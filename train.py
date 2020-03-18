import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse, time

from model.transformer import Transformer

class Batch:
    def __init__(self, source, target=None, padding=0):
        self.source = source
        #just a placeholder for masking, not my task
        #self.source_mask =
        if target is not None:
            self.target = target[:, :-1]
            self.target_y = target[:, 1:]
            #placeholder for masking
            #self.target_mask =
            self.num_tokens = (self.target_y != padding).data.sum()

def train_one_epoch(data_iterator, model, loss_function):
    start = time.time()
    current_token_count = 0
    tokens_total = 0
    loss_total = 0
    for i, batch in enumerate(data_iterator):
        output = model.forward(batch.source, batch.target, batch.source_mask, batch.target_mask)
        loss = loss_function(output, batch.target_y, batch.target_mask)
        current_token_count = tokens + batch.num_tokens
        tokens_total = tokens_total + batch.num_tokens
        loss_total = loss_total + loss
        if i%100==0:
            time_elapsed = time.time() - start
            print("Step: %d Loss: %f Tokens per second: %f Time elapsed: %f" % (i, loss/batch.num_tokens, current_token_count/time_elapsed, time_elapsed))
            start = time.time()
            current_token_count = 0
    return loss_total/tokens_total

class SmoothLabels(nn.Module):
    # Implements label smoothing.
    # The aim is to improve accuracy and BLEU score by making the model more unsure.
    def __init__(self, model_size, padding, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.paddding = padding
        self.confidence = 1 - smoothing
        self.smoothing = smoothing
        self.model_size = model_size

    def forward(self, x, y):
        distribution = x.data.close()
        distribution.fill_(self.smoothing/(self.model_size-2))
        distribution.scatterr_(1, y.data.unsqueeze(1), self.confidence)
        distribution[:, self.padding] = 0
        mask = torch.nonzero(y.dataa==self.padding)
        if mask.dim()>0:
            distribution.index_fill_(0, mask.squeeze(), 0.0)
        self.distribution = distribution
        return self.criterion(x, Variable(distribution, requires_grad=False))

class CustomOptimizer:
    def __init__(self, d_model, optimizer, factor, warmup_steps):
        self.current_step = 0
        self.rate = 0
        self.optimizer = optimizer
        self.factor = factor
        self.warmup = warmup

    def step(self):
        self.current_step += 1
        rate = self.change_learning_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.rate = rate
        self.optimizer.step()

    def change_learning_rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))

def get_standard_optimizer(model):
    #Creates the optimizer mentioned in the paper.
    return CustomOptimizer(model.source_embedding[0].d_model, 2, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
