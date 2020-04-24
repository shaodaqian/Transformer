import torch
import argparse
import math
import time
import dill as pickle
from tqdm import tqdm
import torch.optim as optim

from model.Optim import ScheduledOptim
from torchtext.data import Field, Dataset, BucketIterator
from model.transformer import build_transformer
from model.translator import Translator
from special_tokens import PAD_WORD,BOS_WORD,EOS_WORD,UNK_WORD

from data_process import load_data_dict
import special_tokens
from nltk.translate.bleu_score import sentence_bleu


from special_tokens import PAD_WORD, EOS_WORD, BOS_WORD

def load_model(opt, device):

    checkpoint = torch.load(opt.model, map_location=device)
    model_opt = checkpoint['settings']

    model = build_transformer(
        model_opt.src_vocab_size,
        model_opt.trg_vocab_size,

        src_pad_idx=model_opt.src_pad_idx,
        trg_pad_idx=model_opt.trg_pad_idx,
        d_model=model_opt.d_model,
        d_ff=model_opt.d_inner_hid,
        num_layers=model_opt.n_layers,
        num_attention_layers=model_opt.n_head,
        dropout=model_opt.dropout).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model


def translation_score(pred_seq,ends,gold,TRG):
    bleu=[0,0]
    # bleu[0] is total bleu score, bleu[1] number of sentences
    # print(pred_seq.shape[0],gold.shape[0])
    for i in range(pred_seq.shape[0]):
        current=pred_seq[i][:ends[i]]
        pred_line = ' '.join(TRG.vocab.itos[idx] for idx in current)
        pred_line = pred_line.replace(special_tokens.BOS_WORD, '').replace(special_tokens.EOS_WORD, '').replace(special_tokens.PAD_WORD, '')
        target=gold[i]
        target_line = ' '.join(TRG.vocab.itos[idx] for idx in target)
        target_line = target_line.replace(special_tokens.BOS_WORD, '').replace(special_tokens.EOS_WORD, '').replace(special_tokens.PAD_WORD, '')
        bleu[0]+=sentence_bleu([list(target_line)],list(pred_line))
        bleu[1]+=1
    return bleu

def main():
    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model weight file')
    parser.add_argument('-data_pkl', required=True,
                        help='Pickle file with both instances and vocabulary.')
    parser.add_argument('-test_data', default=None)  # bpe encoded data

    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=4)
    parser.add_argument('-max_seq_len', type=int, default=100)
    parser.add_argument('-no_cuda', action='store_true')

    opts = parser.parse_args()
    opts.cuda = not opts.no_cuda
    device = torch.device('cuda' if opts.cuda else 'cpu')

    data = pickle.load(open(opts.data_pkl, 'rb'))
    SRC, TRG = data['vocab']['src'], data['vocab']['trg']
    opts.src_pad_idx = SRC.vocab.stoi[PAD_WORD]
    opts.trg_pad_idx = TRG.vocab.stoi[PAD_WORD]
    opts.trg_bos_idx = TRG.vocab.stoi[BOS_WORD]
    opts.trg_eos_idx = TRG.vocab.stoi[EOS_WORD]
    unk_idx = SRC.vocab.stoi[SRC.unk_token]

    test_loader = Dataset(examples=data['test'], fields={'src': SRC, 'trg': TRG})

    translator = Translator(
        model=load_model(opts, device),
        beam_size=opts.beam_size,
        max_seq_len=opts.max_seq_len,
        src_pad_idx=opts.src_pad_idx,
        trg_pad_idx=opts.trg_pad_idx,
        trg_bos_idx=opts.trg_bos_idx,
        trg_eos_idx=opts.trg_eos_idx,
        device=device).to(device)

    total_bleu, total_sentence = 0, 0
    for example in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
        # print(' '.join(example.src))
        src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example.src]
        trg_seq=[TRG.vocab.stoi.get(word, unk_idx) for word in example.trg]
        gold=torch.LongTensor([trg_seq[1:]]).to(device)
        pred_seq, ends = translator.translate_sentence(torch.LongTensor([src_seq]).to(device))
        bleu = translation_score(pred_seq, ends, gold, TRG)
        total_bleu += bleu[0]
        total_sentence += bleu[1]
    bleu_score = total_bleu / total_sentence
    print('BLEU score for model: ',bleu_score)
