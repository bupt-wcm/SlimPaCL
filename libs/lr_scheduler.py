import math
from collections import Counter

import numpy as np
import torch
import torch.nn as nn


class WarmCosineLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, warm_steps, base_lr, min_lr, epochs, batch_num):
        self.optimizer = optimizer
        self.warm_steps = warm_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.epochs = epochs
        self.batch_num = batch_num

    def get_learning_rate(self, step):
        if step < self.warm_steps:
            mul = step * 1.0 / self.warm_steps
        else:
            mul = np.cos(
                (step - self.warm_steps) / (self.epochs * self.batch_num - self.warm_steps) * math.pi) * 0.5 + 0.5
        return mul * (self.base_lr - self.min_lr) + self.min_lr

    def step(self, step_num=None):
        current_lr = self.get_learning_rate(step_num)
        for param in self.optimizer.param_groups:
            param['lr'] = current_lr
        return current_lr


class WarmStepLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, warm_steps, base_lr, step_size, gamma):
        self.optimizer = optimizer
        self.warm_steps = warm_steps
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_learning_rate(self, step):
        if step < self.warm_steps:
            mul = step * 1.0 / self.warm_steps
        else:
            mul = (step - self.warm_steps) // self.step_size
            mul = self.gamma ** mul
        return mul * self.base_lr

    def step(self, step_num=None):
        current_lr = self.get_learning_rate(step_num)
        for param in self.optimizer.param_groups:
            param['lr'] = current_lr
        return current_lr


class WarmMultiStonesScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, warm_steps, base_lr, stones, gamma):
        self.optimizer = optimizer
        self.warm_steps = warm_steps
        self.milestones = Counter(stones)
        self.gamma = gamma

        self.base_lr = None
        if base_lr is None:
            self.base_lr = [group['lr'] for group in self.optimizer.param_groups]
        else:
            if isinstance(base_lr, int) or isinstance(base_lr, float):
                self.base_lr = []
                for i in range(len(self.optimizer.param_groups)):
                    self.base_lr.append(base_lr)
            if isinstance(base_lr, list):
                self.base_lr = base_lr
        assert self.base_lr is not None and len(self.base_lr) == len(self.optimizer.param_groups)

    def get_learning_rate(self, step_num):
        if step_num < self.warm_steps:
            mul = step_num * 1.0 / self.warm_steps
            for idx, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.base_lr[idx] * mul

        step_num = step_num - self.warm_steps

        if step_num not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[step_num]
                for group in self.optimizer.param_groups]

    def step(self, step_num=None):
        current_lr = self.get_learning_rate(step_num)
        for idx, param in enumerate(self.optimizer.param_groups):
            param['lr'] = current_lr[idx]
        return current_lr[0]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plots = []
    m = nn.Linear(10, 10)
    optim = torch.optim.SGD(m.parameters(), lr=0.1)
    sched = WarmCosineLRScheduler(optim, 1000, 0.01, 1e-4, 200, 1000)
    cnt = 0
    for i in range(200):
        for j in range(1000):
            lr = sched.step(cnt)
            plots.append(lr)
            cnt += 1
    print(plots)
    plt.plot(plots)
    plt.show()
