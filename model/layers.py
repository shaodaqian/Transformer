import torch.nn as nn
import torch.nn.functional as F

import torch
import numpy as np
import math
from torch.autograd import Variable



class EncoderLayer(nn.Module):
    def __init__(self, attention, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = attention
        self.pos_ffn = feed_forward

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_attn



class DecoderLayer(nn.Module):
    def __init__(self, attn1, attn2, ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = attn1
        self.enc_attn = attn2
        self.ff = ff

    def forward(self, dec_input, enc_output,slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.ff(dec_output)
        return dec_output, dec_attn, dec_enc_attn

class PositionalEncoding(nn.Module):
    def __init__(self,d_model, dropout=0.1,max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
        return self.dropout(x)

class Embeddings(nn.Module):
    def __init__(self,vocab,d_model,mask=None):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(vocab, d_model,mask)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)