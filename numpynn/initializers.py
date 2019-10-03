import numpy as np


class Zeros:
    def __call__(self, shape):
        return np.zeros(shape=shape)


class RandomNormal:
    """Random samples from a normal (Gaussian) distribution.

    Args:
        loc (float or array_like of floats):
            Mean (“centre”) of the distribution.
        scale (float or array_like of floats):
            Standard deviation (spread or “width”) of the distribution.
    """

    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale

    def __call__(self, shape):
        return np.random.normal(loc=self.loc, scale=self.scale, size=shape)


class RandomUniform:
    """Random samples from a uniform distribution.

    Args:
        low (float or array_like of floats):
            Lower boundary of the output interval.
        high (float or array_like of floats):
            Upper boundary of the output interval.
    """

    def __init__(self, low=0.0, high=1.0):
        self.low = low
        self.high = high

    def __call__(self, shape):
        return np.random.uniform(low=self.low, high=self.high, size=shape)

