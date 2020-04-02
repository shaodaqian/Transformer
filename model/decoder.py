import torch
import torch.nn as nn
import numpy as np
from model.sublayers import make_clones, Norm, SublayerConnectionNormalisation, FeedForward, MultiHeadAttention
from model.layers import Embedder, PositionalEncoder


def decoder_mask(d_model):
    shape_attention = (1, d_model, d_model)
    decoder_mask = np.triu(np.ones(shape_attention), k=1).astype('uint8')
    return torch.from_numpy(decoder_mask) == 0

class DecoderLayer(nn.Module):
    "A decoder layer is made up of three sublayers."
    "The first is a multihead self-attention layer, the second one is a multihead source-attention layer,
    "and the third one is a fully-connected feedforward layer."
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        self.sublayers= make_clones(SublayerConnectionNormalisation(d_model, dropout), 3)
        self.attention = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.encoder_attention = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x,memory, encoder_mask, decoder_mask):
        x = self.sublayers[0](x, lambda x: self.attention(x,x,x,decoder_mask))
        x = self.sublayers[1](x, lambda x: self.encoder_attention(x,memory,memory,encoder_mask))
        x = self.sublayers[2](x, lambda x: self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, d_ff, dropout):
        super().__init__()
        self.layer = DecoderLayer(d_model, d_ff, num_heads, dropout=dropout)
        self.layers = make_clones(self.layer, num_layers)
        self.num_layers = num_layers
        self.norm = Norm(d_model)

    "Pass the input through each decoder layer in turn."
    def forward(self, x, encoder_outputs, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_outputs, source_mask, target_mask)
        return self.norm(x)
