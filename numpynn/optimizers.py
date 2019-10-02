import numpy as np
from tqdm import tqdm

from .losses import *
from .activations import *
from .layers import *


class SGD:
    """Stochatic Gradient Descent"""

    def __init__(self, lr=0.01):
        self.lr = lr

    def optimize(self, model, x, y, batch_size):
        """1 epoch optimization."""
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
                losses[bid] += model.layers[l].regularization()

            # Backpropagate
            model.outputs.errors = self.ouput_errors(model, y)
            for l in range(len(model.layers) - 2, 0, -1):
                model.layers[l].backward()

            # Gradient Descent
            m = x.shape[-1]

            for l in range(1, len(model.layers)):
                if isinstance(model.layers[l], Dropout):
                    continue
                
                model.layers[l].update(self.lr, m)

            p_batches.set_description("Batches")
        return np.mean(losses)

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
