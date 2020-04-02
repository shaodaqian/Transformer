import torch
import torch.nn as nn
import numpy as np

from model.sublayers import make_clones, Norm, SublayerConnectionNormalisation, FeedForward, MultiHeadAttention
from model.layers import Embedder, PositionalEncoder


class EncoderLayer(nn.Module):
    "An encoder layer is made up of two sublayers."
    "The first is a multihead self-attention layer and the second one is a" 
    "fully-connected feedforward layer."
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        self.sublayers= make_clones(SublayerConnectionNormalisation(d_model, dropout), 2)
        self.attention = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x, encoder_mask):
        x = self.sublayers[0](x, lambda x: self.attention(x,x,x,encoder_mask))
        x = self.sublayers[1](x, lambda x: self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, d_ff, dropout):
        super().__init__()
        self.layer = EncoderLayer(d_model, d_ff, num_heads, dropout=dropout)
        self.layers = make_clones(self.layer, num_layers)
        self.num_layers = num_layers
        self.norm = Norm(d_model)

    def forward(self, source, mask):
        "Pass the input through each encoder layer in turn."
        for layer in self.layers:
            source = layer(source, mask)
        return self.norm(source)
