import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

def  make_clones(layer, num_layers):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        norm = self.alpha * (x-mean) / (std + self.eps) + self.bias
        return norm

class SublayerConnectionNormalisation(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = Norm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_function):
        return x + self.dropout(sublayer_function(self.norm(x)))

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
         super().__init__()
         self.linear_1 = nn.Linear(d_model, d_ff)
         self.dropout = nn.Dropout(dropout)
         self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropoutt(F.relu(self.linear_1(x))))


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
        # q(b*head*lenq*d_k)  k_t(b*head*d_k*lenk)
        score = torch.matmul(q / math.sqrt(self.d_k), k.transpose(2, 3))
        # score(b*head*lenq*lenk)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(score, dim=-1))
        # v(b*head*lenv*d_v) (lenv=lenk)
        output = torch.matmul(attn, v)
        # output(b*head*lenk*d_v)
        return output, attn

    def forward(self,q,k,v,mask=None):
        d_k, d_v, h = self.d_k, self.d_v, self.h
        sz_b= q.size(0)

        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, -1, h, d_k)
        k = self.w_ks(k).view(sz_b, -1, h, d_k)
        v = self.w_vs(v).view(sz_b, -1, h, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            # mask(b*1*1*len) or (b*1*len*len)
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)

        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, -1, h*d_v)
        q = self.dropout(self.fc(q))
        return q