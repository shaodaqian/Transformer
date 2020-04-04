import torch
import argparse
import math
import time
import dill as pickle
from tqdm import tqdm
import torch.optim as optim

from model.transformer import Transformer,build_transformer
from model.Optim import ScheduledOptim
from torchtext.data import Field, Dataset, BucketIterator

from train import train_one_epoch

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

def dataloaders(opt, device):
    batch_size = opt.batch_size
    data = pickle.load(open(opt.data_pkl, 'rb'))

    opt.max_token_seq_len = data['settings'].max_len
    opt.src_pad_idx = data['vocab']['src'].vocab.stoi[PAD_WORD]
    opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi[PAD_WORD]

    opt.src_vocab_size = len(data['vocab']['src'].vocab)
    opt.trg_vocab_size = len(data['vocab']['trg'].vocab)

    # if opt.embs_share_weight:
    #     assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
    #         'To sharing word embedding the src/trg word2idx table shall be the same.'

    fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg']}

    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    return train_iterator, val_iterator

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_pkl', default=None)  # all-in-1 data pickle or bpe field

    parser.add_argument('-train_path', default=None)  # bpe encoded data
    parser.add_argument('-val_path', default=None)  # bpe encoded data

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup', '--warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    args.d_word_vec = args.d_model

    if not args.log and not args.save_model:
        print('No experiment result will be saved.')
        raise ValueError

    if args.batch_size < 2048 and args.warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n' \
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n' \
              'Using smaller batch w/o longer warmup may cause ' \
              'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if args.cuda else 'cpu')

    # ========= Loading Dataset =========#

    if args.data_pkl:
        training_data, validation_data = dataloaders(args, device)
    else:
        raise NotADirectoryError(args.data_pkl)

    print(args)
    transformer = build_transformer(
        args.src_vocab_size,
        args.trg_vocab_size,
        src_pad_idx=args.src_pad_idx,
        trg_pad_idx=args.trg_pad_idx,
        d_model=args.d_model,
        d_ff=args.d_inner_hid,
        num_layers=args.n_layers,
        num_attention_layers=args.n_head,
        dropout=args.dropout
    ).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-09),2.0, args.d_model, args.warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device, args)

if __name__ == "__main__":
    main()