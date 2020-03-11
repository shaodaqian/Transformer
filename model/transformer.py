import torch
import torch.nn as nn
import numpy as np
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module, encoder, decoder, output_generator, source_embedding, target_embedding):
    def __init__(self, encoder, decoder, source_vocab, target_vocab, output_generator):
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
    encoder = Encoder(num_layers, num_heads, vocab_size, d_model, d_ff, dropout=dropout)
    decoder = Decoder(num_layers, num_heads, vocab_size, d_model, d_ff, dropout=dropout)
    source_embedding = nn.Sequantial(Embeddings(d_model, source_vocab), copy.deepcopy(positional_encoder))
    target_embedding = nn.Sequantial(Embeddings(d_model, target_vocab), copy.deepcopy(positional_encoder))
    generator = OutputGenerator(d_model, target_vocab)
    return Transformer()
