import torch
import torch.nn as nn
import numpy as np

from sublayers import make_clones, Norm, SublayerConnectionNormalisation, FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        self.sublayers= make_clones(SublayerConnectionNormalisation(d_model, dropout), 2)
        self.attention = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x, encoder_mask):
        x = self.sublayers[0](x, lambda x: self.attention(x,x,x,mask))
        x = self.sublayers[1](x, lambda x: self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, num_heads, size_vocab, d_model, d_ff, dropout):
        super().__init__()
        self.layer = EncoderLayer(d_model, d_ff, num_heads, dropout=dropout)
        self.layers = make_clones(self.layer, num_layers)
        self.num_layers = num_layers
        self.embedding = Embedder(size_vocab, d_model)
        self.positional_encoder = PositionalEncoder(d_model, dropout=dropout)
        self.norm = Norm(d_model)

    def forward(self, source, mask):
        x = self.embedding(source)
        x = self.positional_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
