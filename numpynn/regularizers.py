import numpy as np


class L2:
    def __init__(self, lmbda):
        self.lmbda = lmbda

    def __call__(self, all_weights):
        if self.lmbda == 0:
            return 0
        sum_squared_weights = np.sum([np.sum(w ** 2) for w in all_weights])
        return self.lmbda * sum_squared_weights

    def shrink(self, lr, weights):
        if self.lmbda == 0:
            return weights
        return weights * (1 - lr * self.lmbda * 2)


class L1:
    def __init__(self, lmbda):
        self.lmbda = lmbda

    def __call__(self, all_weights):
        if self.lmbda == 0:
            return 0
        sum_abs_weights = np.sum([np.sum(abs(w)) for w in all_weights])
        return self.lmbda * sum_abs_weights

    def shrink(self, lr, weights):
        if self.lmbda == 0:
            return weights
        return weights - lr * self.lmbda * np.sign(weights)
