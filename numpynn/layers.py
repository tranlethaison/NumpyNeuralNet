import numpy as np

from .activations import Linear
from .initializers import Zeros, RandomNormal


class Input:
    def __init__(self, units):
        self.units = units

    def forward(self, x):
        self.activations = x


class Base:
    """Base layer for others to inherit from, except for `Input`."""

    def __call__(self, prior_layer):
        self.prior_layer = prior_layer
        self.prior_layer.next_layer = self
        return self


class Dense(Base):
    def __init__(
        self,
        units,
        activation=Linear,
        kernel_initializer=RandomNormal(),
        bias_initializer=Zeros(),
        kernel_regularizer=None,
    ):
        self.units = units
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer

    def init_bias(self):
        self.bias = np.array(
            [self.bias_initializer(shape=(1)) for _ in range(self.units)]
        )

    def init_weights(self):
        self.weights = np.array(
            [
                self.kernel_initializer(shape=(self.prior_layer.units))
                for _ in range(self.units)
            ]
        )

    def forward(self, **kwargs):
        self.affines = np.matmul(self.weights, self.prior_layer.activations) + self.bias
        self.activations = self.activation.f(self.affines)

    def backward(self):
        self.errors = np.matmul(
            self.next_layer.weights.T, self.next_layer.errors
        ) * self.activation.df(self.affines)

    def update(self, lr, m):
        if self.kernel_regularizer:
            self.weights = self.kernel_regularizer.shrink(lr, self.weights)

        self.weights -= lr * np.matmul(self.errors, self.prior_layer.activations.T) / m

        self.bias -= lr * np.sum(self.errors, axis=-1, keepdims=True) / m

    def regularization(self):
        if self.kernel_regularizer:
            return self.kernel_regularizer(self.weights)
        return 0


class Dropout(Base):
    def __init__(self, rate=0.5):
        self.rate = rate

    def __call__(self, prior_layer):
        super().__call__(prior_layer)

        self.units = self.prior_layer.units

        n_drops = int(self.units * self.rate)
        self.drops = np.expand_dims(
            np.hstack([np.zeros(n_drops), np.ones(self.units - n_drops)]), axis=-1
        )
        return self

    def forward(self, is_training=False):
        if is_training:
            self.drops = np.random.permutation(self.drops)
            self.activations = self.prior_layer.activations * self.drops
        else:
            self.activations = self.prior_layer.activations * self.rate

    def backward(self):
        self.weights = self.next_layer.weights * self.drops.T
        self.errors = self.next_layer.errors
