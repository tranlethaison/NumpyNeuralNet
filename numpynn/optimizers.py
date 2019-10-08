import numpy as np
from tqdm import tqdm

from .losses import *
from .activations import *
from .layers import *


class SGD:
    """Stochatic Gradient Descent"""

    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        assert 0 <= momentum <=1
        self.momentum = momentum

    def optimize(self, model, x, y, batch_size):
        """1 epoch optimization."""
        self.init_velocities(model)

        # Shuffle data
        ids = np.arange(x.shape[0])
        np.random.shuffle(ids)
        x = x[ids, ...]
        y = y[ids, ...]

        batches = [
            (x[i : i + batch_size, ...], y[i : i + batch_size, ...])
            for i in range(0, len(x), batch_size)
        ]
        losses = [None] * len(batches)

        p_batches = tqdm(batches)
        for bid, (x, y) in enumerate(p_batches):
            x, y = x.T, y.T

            # Feedforward
            model.inputs.forward(x)
            for l in range(1, len(model.layers)):
                model.layers[l].forward(is_training=True)

            # Loss
            losses[bid] = model.loss.f(y, model.outputs.activations)
            for l in range(1, len(model.layers)):
                if not model.layers[l].trainable:
                    continue
                losses[bid] += model.layers[l].regularization()

            # Backpropagate
            model.outputs.errors = self.ouput_errors(model, y)
            for l in range(len(model.layers) - 2, 0, -1):
                model.layers[l].backward()

            # Gradient Descent
            m = x.shape[-1]
            for l in range(1, len(model.layers)):
                if not model.layers[l].trainable:
                    continue
                self.update(model, l, m)

            p_batches.set_description("Batches")
        return np.mean(losses)

    def init_velocities(self, model):
        self.v_weights = [None] * len(model.layers)
        self.v_bias = [None] * len(model.layers)

        for l in range(len(model.layers)):
            if not model.layers[l].trainable:
                continue
            self.v_weights[l] = np.zeros(model.layers[l].weights.shape)
            self.v_bias[l] = np.zeros(model.layers[l].bias.shape)

    def ouput_errors(self, model, y):
        """Returns partial derivative of Loss wrt output affines."""
        if (
            model.loss is CrossEntropy
            and model.outputs.activation is Sigmoid
        ):
            return model.outputs.activations - y

        if (
            model.loss is LogLikelihood
            and model.outputs.activation is Softmax
        ):
            return model.outputs.activations - y

        return (
            model.loss.df_da(y, model.outputs.activations)
            * model.outputs.activation.df(model.outputs.affines)
        )

    def update(self, model, l, m):
        # weights
        dcdw = (
            np.matmul(model.layers[l].errors, model.layers[l].prior_layer.activations.T)
            / m
        )
        self.v_weights[l] = self.momentum * self.v_weights[l] - self.lr * dcdw
        if model.layers[l].kernel_regularizer:
            model.layers[l].weights = model.layers[l].kernel_regularizer.shrink(
                self.lr, model.layers[l].weights
            )
        model.layers[l].weights += self.v_weights[l]

        # bias
        dcdb = np.sum(model.layers[l].errors, axis=-1, keepdims=True) / m
        self.v_bias[l] = self.momentum * self.v_bias[l] - self.lr * dcdb
        model.layers[l].bias += self.v_bias[l]
