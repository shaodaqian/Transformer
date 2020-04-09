class ScheduledOptim:
    "Optim wrapper that implements rate."

    def __init__(self, optimizer, init_lr, model_size, warmup):
        self.optimizer = optimizer
        self.n_step = 0
        self.warmup = warmup
        self.init_lr = init_lr
        self.model_size = model_size

    def step(self):
        "Update parameters and rate"
        self.n_step += 1
        rate = self.rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self.n_step
        return self.init_lr * (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self.optimizer.zero_grad()
