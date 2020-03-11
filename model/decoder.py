import torch
import torch.nn as nn
import numpy as np

def decoder_mask(d_model):
    shape_attention = (1, d_model, d_model)
    decoder_mask = np.triu(np.ones(shape_attention), k=1).astype('uint8')
    return torch.from_numpy(decoder_mask) == 0

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        self.sublayers= make_clones(SublayerConnectionNormalisation(d_model, dropout), 3)
        self.attention = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.encoder_attention = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x, encoder_mask, decoder_mask, memory):
        x = self.sublayers[0](x, lambda x: self.attention(x,x,x,decoder_mask))
        x = self.sublayers[1](x, lambda x: self.encoder_attention(x,memory,memory,encoder_mask))
        x = self.sublayers[2](x, lambda x: self.feed_forward)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, num_heads, vocab_size, d_model, d_ff, dropout):
        super().__init__()
        self.layer = DecoderLayer(d_model, d_ff, num_heads, dropout=dropout)
        self.layers = make_clones(self.layer, num_layers)
        self.num_layers = num_layers
        self.embedding = Embedder(vocab_size, d_model)
        self.positional_encoder = PositionalEncoder(d_model, dropout=dropout)
        self.norm = Norm(layer.size)

    def forward(self, target, encoder_outputs, source_mask, target_mask):
        x = self.embedding(target)
        x = self.positional_encoder(x)
        for i in range(self.num_layers):
            x = self.layers[i](x, encoder_outputs, source_mask, target_mask)
        return self.norm(x)
