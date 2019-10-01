import numpy as np


class L2:
    def __init__(self, lmbda):
        self.lmbda = lmbda

    def __call__(self, n, weights):
        if self.lmbda == 0:
            return 0
        sum_squared_weights = np.sum([np.sum(w) ** 2 for w in weights])
        return (self.lmbda * sum_squared_weights) / (2 * n)

    def weight_scale_factor(self, lr, n):
        return 1 - lr * self.lmbda / n
