import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
import math
from model.translator import Translator, TranslatorParallel
from translate import translation_score
from datetime import datetime
import os


def patch_source(source):
    source = source.transpose(0, 1)
    return source


def patch_target(target):
    target = target.transpose(0, 1)
    gold = target[:, 1:]
    target = target[:, :-1]
    return target, gold


def calculate_metrics(prediction, gold, trg_pad_idx, smoothing=False):
    loss = compute_loss(prediction, gold, trg_pad_idx, smoothing=smoothing)
    prediction = prediction.max(1)[1]
    non_pad_mask = gold.ne(trg_pad_idx)
    num_correct = prediction.eq(gold).masked_select(non_pad_mask).sum().item()
    num_words = non_pad_mask.sum().item()
    return loss, num_correct, num_words


def compute_loss(prediction, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    if smoothing:
        eps = 0.1
        n_class = prediction.size(1)
        one_hot = torch.zeros_like(prediction).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(prediction, dim=1)
        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(prediction, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def run_one_epoch(model, data, args, device, TRG, total_tokens, optimizer=None, smoothing=False, bleu=False):
    ''' Epoch operation in training phase'''
    training = optimizer is not None
    total_loss, total_num_words, total_num_correct_words, total_bleu, total_sentence = 0, 0, 0, 0, 0
    if training:
        desc = '  - (Training)   '
        model.train()
    else:
        desc = '  - (Validation) '
        model.eval()
    if bleu:
        translator = Translator(
            model=model,
            beam_size=args.beam_size,
            max_seq_len=args.max_seq_len,
            src_pad_idx=args.src_pad_idx,
            trg_pad_idx=args.trg_pad_idx,
            trg_bos_idx=args.trg_bos_idx,
            trg_eos_idx=args.trg_eos_idx,
            device=device
        )
        translator = TranslatorParallel(translator)
        # translator = CustomDataParallel(translator)

    for batch in tqdm(data, mininterval=10, desc=desc, leave=False, total=total_tokens//args.batch_size):
        # prepare data
        source_sequence = patch_source(batch.src).to(device)
        target_sequence, gold = map(lambda x: x.to(device), patch_target(batch.trg))
        # source_sequence = nn.DataParallel(patch_source(batch.src))
        # target_sequence, gold = map(lambda x: nn.DataParallel(x), patch_target(batch.trg))
        # forward pass
        if training:
            optimizer.zero_grad()
        if bleu:
            pred_seq, ends = translator.translate_sentence(source_sequence)
            score = translation_score(pred_seq, ends, gold, TRG)
            total_bleu += score[0]
            total_sentence += score[1]
        prediction = model(source_sequence, target_sequence)
        output = model.generator(prediction)
        output = output.view(-1, output.size(-1))
        # backward pass and update parameters
        loss, num_correct, num_words = calculate_metrics(
            output, gold.contiguous().view(-1), args.trg_pad_idx, smoothing=smoothing
        )
        if training:
            loss.backward()
            optimizer.step_and_update_lr()
        total_num_words += num_words
        total_num_correct_words += num_correct
        total_loss += loss.item()
    if total_num_words != 0:
        loss_per_word = total_loss/total_num_words
        accuracy = total_num_correct_words / total_num_words
        if bleu:
            bleu_score = total_bleu / total_sentence
            print('current BLEU score: ', bleu_score)
        else:
            bleu_score = None
        return loss_per_word, accuracy, bleu_score
    else:
        return 0, 0, None


def train(
    model, training_data, validation_data, optimizer, args, device,
    SRC, TRG, bleu_freq, total_training_tokens, total_dev_tokens):
    ''' Start training '''
    log_train_file, log_valid_file = None, None
    "We can optionally log the training and validation processes."
    if args.log:
        log_train_file = f'{args.log}.train.log'
        log_valid_file = f'{args.log}.valid.log'
        print(f'[Info] Training performance will be written to file: {log_train_file} and {log_valid_file}')
        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    "Utility function for printing performance at a given time."
    def get_performance_string(epoch, loss, accu, start_time, bleu=None):
        ppl = math.exp(min(loss, 100))
        elapse = (time.time()-start_time) / 60
        accu = 100*accu
        perf = f'e: {epoch}, ppl: {ppl: 8.3f}, accuracy: {accu:3.2f}%, elapse: {elapse:3.2f} min, time of finish: {datetime.fromtimestamp(time.time())}\n'
        if bleu is not None:
            perf += f'BLEU: {bleu}\n'
        return perf

    for epoch_number in range(args.epoch):
        print('[ Epoch', epoch_number, ']')
        cal_bleu = False
        start_time = time.time()
        training_loss, training_accuracy, _ = run_one_epoch(
            model,
            training_data,
            args,
            device,
            TRG,
            total_tokens=total_training_tokens,
            optimizer=optimizer,
            smoothing=args.label_smoothing
        )
        # start = time.time()
        if epoch_number % bleu_freq == (bleu_freq - 1):
            cal_bleu = True
        validation_loss, validation_accuracy, bleu_score = run_one_epoch(
            model,
            validation_data,
            args,
            device,
            TRG,
            total_tokens=total_dev_tokens,
            optimizer=None,
            bleu=cal_bleu
        )
        if epoch_number == 0:
            min_val_loss = validation_loss
        checkpoint = {'epoch': epoch_number, 'settings': args, 'model': model.state_dict()}
        "Optionally save the model."
        if args.experiment_name:
            if args.save_mode == 'all':
                model_name = f'{args.experiment_name}_accu_{validation_accuracy:3.3f}.chkpt'
                torch.save(checkpoint, model_name)
            elif args.save_mode == 'best':
                model_name = f'{args.experiment_name}.chkpt'
                if validation_loss <= min_val_loss:
                    min_val_loss = validation_loss
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')
            elif args.save_mode == 'last_5':
                model_name = f'{args.experiment_name}-{epoch_number}.chkpt'
                torch.save(checkpoint, model_name)
                if epoch_number >= 5:
                    oldest_model_name = f'{args.experiment_name}-{epoch_number-5}.chkpt'
                    os.remove(oldest_model_name)

        "Optionally log the training/validation step."
        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                results = get_performance_string(epoch_number, training_loss, training_accuracy, start_time)
                log_tf.write(results)
                results = get_performance_string(
                    epoch_number, validation_loss, validation_accuracy, start_time, bleu_score)
                log_vf.write(results)
