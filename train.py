import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse, time
from tqdm import tqdm
import math


def patch_source(source):
    source = source.transpose(0, 1)
    return source


def patch_target(target):
    target = target.transpose(0, 1)
    gold = target[:, 1:].contiguous().view(-1)
    target = target[:, :-1]
    return target, gold


def calculate_metrics(prediction, gold, trg_pad_idx, smoothing=False):
    loss = compute_loss(prediction, gold, trg_pad_idx, smoothing=smoothing)
    prediction = prediction.max(1)[1]
    gold = gold.contiguous().view(-1)
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


def train_one_epoch(model, training_data, optimizer, args, device, smoothing=False):
    ''' Epoch operation in training phase'''
    model.train()
    total_loss, total_num_words, total_num_correct_words = 0, 0, 0
    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):
        # prepare data
        source_sequence = patch_source(batch.src).to(device)
        target_sequence, gold = map(lambda x: x.to(device), patch_target(batch.trg))
        # forward pass
        optimizer.zero_grad()
        prediction = model(source_sequence, target_sequence)
        output=model.generator(prediction)
        output = output.view(-1, output.size(-1))
        # backward pass and update parameters
        loss, num_correct, num_words = calculate_metrics(
            output, gold, args.trg_pad_idx, smoothing=smoothing
        )
        loss.backward()
        optimizer.step_and_update_lr()
        total_num_words += num_words
        total_num_correct_words += num_correct
        total_loss += loss.item()
    if total_num_words != 0:
        loss_per_word = total_loss/total_num_words
        accuracy = total_num_correct_words / total_num_words
    else:
        loss_per_word = 0
        accuracy = 0
    return loss_per_word, accuracy


def eval_one_epoch(model, validation_data, args, device,smoothing=False):
    ''' Epoch operation in evaluation phase '''
    model.eval()
    total_loss, total_num_words, total_num_correct_words = 0, 0, 0
    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
            # prepare data
            source_sequence = patch_source(batch.src).to(device)
            target_sequence, gold = map(lambda x: x.to(device), patch_target(batch.trg))
            # forward
            prediction = model(source_sequence, target_sequence)
            output = model.generator(prediction)
            output = output.view(-1, output.size(-1))
            loss, num_correct, num_words = calculate_metrics(
                output, gold, args.trg_pad_idx, smoothing=smoothing
            )
            # note keeping
            total_num_words += num_words
            total_num_correct_words += num_correct
            total_loss += loss.item()
    if total_num_words != 0:
        loss_per_word = total_loss/total_num_words
        accuracy = total_num_correct_words/total_num_words
    else:
        loss_per_word = 0
        accuracy = 0
    return loss_per_word, accuracy


def train(model, training_data, validation_data, optimizer, args, device):
    ''' Start training '''
    log_train_file, log_valid_file = None, None
    "We can optionally log the training and validation processes."
    if args.log:
        log_train_file = args.log + '.train.log'
        log_valid_file = args.log + '.valid.log'
        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))
        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    "Utility function for printing performance at a given time."
    def print_performances(header, loss, accu, start_time):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", ppl=math.exp(min(loss, 100)),
                  accu=100*accu, elapse=(time.time()-start_time)/60))

    validation_losses = []
    for epoch_number in range(args.epoch):
        print('[ Epoch', epoch_number, ']')
        start_time = time.time()
        training_loss, training_accuracy = train_one_epoch(
            model,
            training_data,
            optimizer,
            args,
            device,
            smoothing=args.label_smoothing
        )
        print_performances('Training', training_loss, training_accuracy, start_time)
        # start = time.time()
        validation_loss, validation_accuracy = eval_one_epoch(
            model,
            validation_data,
            args,
            device
        )
        print_performances('Validation', validation_loss, validation_accuracy, start_time)
        validation_losses += [validation_loss]
        checkpoint = {'epoch': epoch_number, 'settings': args, 'model': model.state_dict()}
        "Optionally save the model."
        if args.save_model:
            if args.save_mode == 'all':
                model_name = args.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*validation_accuracy)
                torch.save(checkpoint, model_name)
            elif args.save_mode == 'best':
                model_name = args.save_model + '.chkpt'
                if validation_loss <= min(validation_losses):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        "Optionally log the training/validation step."
        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_number, loss=training_loss,
                    ppl=math.exp(min(training_loss, 100)), accu=100*training_accuracy))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_number, loss=validation_loss,
                    ppl=math.exp(min(validation_loss, 100)), accu=100*validation_accuracy))
