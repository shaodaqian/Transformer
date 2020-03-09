import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.h = h
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, h * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, h * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, h * d_v, bias=False)
        self.fc = nn.Linear(h * d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def attention(self,q,k,v, mask=None):
        score = torch.matmul(q / math.sqrt(self.d_k), k.transpose(2, 3))

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(score, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

    def forward(self,q,k,v,mask=None):
        d_k, d_v, h = self.d_k, self.d_v, self.h
        sz_b= q.size(0)

        residual = q
        q = self.layer_norm(q)

        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, -1, h, d_k)
        k = self.w_ks(k).view(sz_b, -1, h, d_k)
        v = self.w_vs(v).view(sz_b, -1, h, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)

        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, -1, h*d_v)
        q = self.dropout(self.fc(q))
        q += residual
        return q, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_inner,dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_inner) # position-wise
        self.w_2 = nn.Linear(d_inner, d_model) # position-wise
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(self.dropout1(F.relu(self.w_1(x))))
        x = self.dropout2(x)
        x += residual
        return x