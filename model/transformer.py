import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.encoder import Encoder
from model.decoder import Decoder
from model.layers import Embedder, PositionalEncoder
import copy


def src_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1)


def trg_mask(seq, pad_idx):
    ''' For masking out the subsequent info. '''
    src = src_mask(seq, pad_idx)
    sz_b, len_s = seq.size()
    utri=torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)
    subsequent_mask = (1 - utri).bool()
    return src & subsequent_mask


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, output_generator, source_embedding, target_embedding, trg_pad_idx, src_pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.generator = output_generator
        self.trg_pad_idx=trg_pad_idx
        self.src_pad_idx=src_pad_idx

    def encode(self, source, source_mask):
        embedding = self.source_embedding(source)
        return self.encoder(embedding, source_mask)

    def decode(self, target, memory, source_mask, target_mask):
        return self.decoder(self.target_embedding(target), memory, source_mask, target_mask)

    def forward(self, source, target):
        self.source_mask = src_mask(source, self.src_pad_idx)
        self.target_mask = trg_mask(target, self.trg_pad_idx)
        encoding = self.encode(source, self.source_mask)
        decoding = self.decode(target, encoding, self.source_mask, self.target_mask)
        return decoding


class OutputGenerator(nn.Module):
    def __init__(self, decoding_output, vocab):
        super().__init__()
        self.projection = nn.Linear(decoding_output, vocab)

    def forward(self, x):
        return F.log_softmax(self.projection(x), dim=1)


def build_transformer(source_vocab, target_vocab, trg_pad_idx, src_pad_idx, num_layers=6, num_attention_layers=8, d_model=512, d_ff=2048, dropout=0.1):
    # we can do a shared vocab here to share the weights for two embeddings and outputgenerator projection
    positional_encoder = PositionalEncoder(d_model, dropout)
    encoder = Encoder(num_layers, num_attention_layers, d_model, d_ff, dropout=dropout)
    decoder = Decoder(num_layers, num_attention_layers, d_model, d_ff, dropout=dropout)
    source_embedding = nn.Sequential(Embedder(source_vocab, d_model,src_pad_idx), copy.deepcopy(positional_encoder))
    target_embedding = nn.Sequential(Embedder(target_vocab, d_model,trg_pad_idx), copy.deepcopy(positional_encoder))
    generator = OutputGenerator(d_model, target_vocab)
    model = Transformer(encoder,decoder,generator,source_embedding,target_embedding,trg_pad_idx,src_pad_idx)
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform(p)
    return model
