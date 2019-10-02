import numpy as np


class L2:
    def __init__(self, lmbda):
        self.lmbda = lmbda

    def __call__(self, weights):
        if self.lmbda == 0:
            return 0
        return self.lmbda * np.sum(np.square(weights))

    def shrink(self, lr, weights):
        if self.lmbda == 0:
            return weights
        return weights * (1 - lr * self.lmbda * 2)


class L1:
    def __init__(self, lmbda):
        self.lmbda = lmbda

    def __call__(self, weights):
        if self.lmbda == 0:
            return 0
        return self.lmbda * np.sum(np.abs(weights))

    def shrink(self, lr, weights):
        if self.lmbda == 0:
            return weights
        return weights - lr * self.lmbda * np.sign(weights)
