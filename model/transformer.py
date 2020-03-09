import torch.nn as nn
import torch
import numpy as np
import copy
from model.layers import EncoderLayer, DecoderLayer, PositionalEncoding, Embeddings
from model.sublayers import MultiHeadAttention, PositionwiseFeedForward
import math


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, n_layers, attn, ff, d_model, embedding, position, dropout=0.1):
        super().__init__()
        self.word_emb = embedding
        self.position_enc = position
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            copy.deepcopy(EncoderLayer(attn, ff))
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False):
        enc_slf_attn_list = []

        enc_output = self.dropout(self.position_enc(self.word_emb(src_seq)*math.sqrt(self.d_model)))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, n_layers, attn1, attn2, ff,d_model, embedding, position, dropout=0.1):
        super().__init__()

        self.word_emb = embedding
        self.position_enc = position
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            copy.deepcopy(DecoderLayer(attn1, attn2, ff))
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = self.dropout(self.position_enc(self.word_emb(trg_seq)*math.sqrt(self.d_model)))

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        dec_output = self.layer_norm(dec_output)

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, max_len=2000,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True):

        super().__init__()
        c = copy.deepcopy
        attn = MultiHeadAttention(n_head, d_model,d_k,d_v,dropout)
        ff = PositionwiseFeedForward(d_model, d_inner, dropout)
        enc_emb=nn.Embedding(vocab, d_word_vec, src_pad_idx)
        dec_emb=nn.Embedding(vocab, d_word_vec, trg_pad_idx)
        position=PositionalEncoding(d_word_vec, max_len=max_len)

        enc_emb.weight=dec_emb.weight

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        self.encoder = Encoder(d_model=d_model,n_layers=n_layers, dropout=dropout,
            atten=c(attn), ff=c(ff),position=c(position),embedding=enc_emb)

        self.decoder = Decoder(d_model=d_model, n_layers=n_layers,
            dropout=dropout,attn1=c(attn),attn2=c(attn),
                        ff=c(ff),position=c(position),embedding=dec_emb)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        self.trg_word_prj.weight=enc_emb.weight
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,src,trg):
        src_mask = get_pad_mask(src, self.src_pad_idx)
        trg_mask = get_pad_mask(trg, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src, src_mask)
        dec_output, *_ = self.decoder(trg, trg_mask, enc_output, src_mask)
        logit=self.trg_word_prj(dec_output)

        return logit.view(-1, logit.size(-1))
