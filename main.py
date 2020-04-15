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
from data_process import load_data_dict

from train import train

from special_tokens import PAD_WORD


def dataloaders(opt, device):
    batch_size = opt.batch_size
    data = load_data_dict(opt)
    opt.max_token_seq_len = data['max_len']
    print(data['train'].fields)
    opt.src_pad_idx = data['train'].fields['src'].vocab.stoi[PAD_WORD]
    opt.trg_pad_idx = data['train'].fields['trg'].vocab.stoi[PAD_WORD]
    opt.src_vocab_size = len(data['train'].fields['src'].vocab)
    opt.trg_vocab_size = len(data['train'].fields['trg'].vocab)
    train = data['train']
    val = data['valid']
    # Turn data into iterators for memory efficiency
    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True, sort=False, shuffle=False)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device, sort=False, shuffle=False)
    return train_iterator, val_iterator


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_pkl', default=None)  # all-in-1 data pickle or bpe field

    parser.add_argument('-train_data', default=None)  # bpe encoded data
    parser.add_argument('-val_data', default=None)  # bpe encoded data

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)

    parser.add_argument('-d_model', type=int, default=128)
    parser.add_argument('-d_inner_hid', type=int, default=128)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-warmup', '--warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true', default=False)
    parser.add_argument('-label_smoothing', action='store_true',default=True)

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    args.d_word_vec = args.d_model


    # if not args.log and not args.save_model:
    #     print('No experiment result will be saved.')
    #     raise ValueError('No save location given')

    if args.batch_size < 2048 and args.warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n' \
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n' \
              'Using smaller batch w/o longer warmup may cause ' \
              'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if args.cuda else 'cpu')

    # ========= Loading Dataset =========#

    training_data, validation_data = dataloaders(args, device)

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
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        2.0,
        args.d_model,
        args.warmup_steps
    )

    train(
        transformer,
        training_data=training_data,
        validation_data=validation_data,
        optimizer=optimizer,
        args=args,
        device=device
    )

if __name__ == "__main__":
    main()