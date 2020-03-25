import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.encoder import Encoder
from model.decoder import Decoder
from model.layers import Embedder, PositionalEncoder
import copy


def src_mask(seq, pad_idx):
    print(pad_idx)
    return (seq != pad_idx).unsqueeze(1)


def trg_mask(seq,pad_idx):
    ''' For masking out the subsequent info. '''
    src=src_mask(seq,pad_idx)
    sz_b, len_s = seq.size()
    utri=torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)
    subsequent_mask = (1 - utri).bool()
    return src & subsequent_mask


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, output_generator, source_embedding, target_embedding):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.output_generator = output_generator
    def encode(self, source, source_mask):

        return self.encoder(self.source_embedding(source), source_mask)

    def decode(self, target, source_mask, target_mask, memory):
        return self.decoder(self.target_embedding(target), source_mask, target_mask, memory)

    def forward(self, source, target, source_mask, target_mask):
        return self.decode(self.encode(source, source_mask), target, source_mask, target_mask)

class OutputGenerator(nn.module):
    def __init__(self, decoding_output, vocab):
        super().__init__()
        self.projection = nn.Linear(decoding_output, vocab)

    def forward(self, x):
        return F.log_softmax(self.projection(x), dim=1)

def build_transformer(source_vocab, target_vocab, num_layers=6, num_attention_layers=8, d_model=512, d_ff=2048, dropout=0.1):
    positional_encoder = PositionalEncoder(d_model, dropout)
    encoder = Encoder(num_layers, num_attention_layers, d_model, d_ff, dropout=dropout)
    decoder = Decoder(num_layers, num_attention_layers, d_model, d_ff, dropout=dropout)
    source_embedding = nn.Sequantial(Embedder(source_vocab, d_model), copy.deepcopy(positional_encoder))
    target_embedding = nn.Sequantial(Embedder(target_vocab, d_model), copy.deepcopy(positional_encoder))
    generator = OutputGenerator(d_model, target_vocab)
    return Transformer(encoder,decoder,generator,source_embedding,target_embedding)
