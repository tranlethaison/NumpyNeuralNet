import numpy as np

from .activations import Linear
from .initializers import Zeros, RandomNormal


class Input:
    """Input layer."""

    def __init__(self, units):
        self.units = units


class Base:
    """Base layer for others to inherit from."""

    def __call__(self, prior_layer):
        self.prior_layer = prior_layer
        self.prior_layer.next_layer = self
        return self

    def init_bias(self):
        pass

    def init_weights(self):
        pass


class Dense(Base):
    """Densely connected layer."""

    def __init__(
        self,
        units,
        activation=Linear,
        kernel_initializer=RandomNormal(),
        bias_initializer=Zeros(),
    ):
        self.units = units
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def init_bias(self):
        self.bias = self.bias_initializer(shape=(self.units, 1), dtype=np.float64)

    def init_weights(self):
        self.weights = self.kernel_initializer(
            shape=(self.units, self.prior_layer.units), dtype=np.float64
        )

    def forward(self, is_training=False):
        self.affines = (
            np.matmul(sefl.weights, self.prior_layer.activations) + self.bias
        )
        self.activations = self.activation.f(self.affines)

    def backward(self):
        self.errors = (
            np.matmul(self.next_layer.weights.T, self.next_layer.errors)
            * self.activation.df(self.affines)
        )


class Dropout(Base):
    """Dropout layer."""

    def __init__(self, rate=0.5):
        self.rate = rate

        pu = self.prior_layer.units
        n_drops = int(pu * rate)
        self.drops = np.expand_dims(
            np.hstack([np.zeros(n_drops), np.ones(pu - n_drops)]), axis=-1
        )
        
    def forward(self, is_training=False):
        if is_training:
            self.drops = np.random.permutation(self.drops)
            self.activations = self.prior_layer.activations * self.drops
        else:
            self.activations = self.prior_layer.activations * self.rate

    def backward(self):
        self.errors = self.prior_layer.errors * self.drops
