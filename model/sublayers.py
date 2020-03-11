import torch
import torch.nn as nn
import torch.nn.functional as F

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
