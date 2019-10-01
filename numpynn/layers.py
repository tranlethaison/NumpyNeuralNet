import numpy as np

from .activations import Linear
from .initializers import Zeros, RandomNormal


class Dense:
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

    def __call__(self, prior_layer):
        self.prior_layer = prior_layer
        return self

    def init_bias(self):
        self.bias = self.bias_initializer(shape=(self.units, 1), dtype=np.float64)

    def init_weights(self):
        self.weights = self.kernel_initializer(
            shape=(self.units, self.prior_layer.units), dtype=np.float64
        )
