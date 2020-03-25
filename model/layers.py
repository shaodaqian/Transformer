import torch.nn as nn
import torch.nn.functional as F

import torch
import numpy as np
import math
from torch.autograd import Variable



class PositionalEncoder(nn.Module):
    def __init__(self,d_model, dropout=0.1,max_len=2000):
        super(PositionalEncoder, self).__init__()
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

class Embedder(nn.Module):
    def __init__(self,vocab,d_model,mask=None):
        super(Embedder, self).__init__()
        self.emb = nn.Embedding(vocab, d_model,mask)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)