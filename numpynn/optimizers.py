import numpy as np
from tqdm import tqdm

from .losses import *
from .activations import *


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

        n = len(x)

        batches = [
            (x[i : i + batch_size, ...], y[i : i + batch_size, ...])
            for i in range(0, len(x), batch_size)
        ]
        losses = [None] * len(batches)

        p_batches = tqdm(batches)
        for bid, (x, y) in enumerate(p_batches):
            x, y = x.T, y.T

            # Feedforward
            z = [None] * len(model.layers)
            a = [None] * len(model.layers)

            a[0] = x
            for l in range(1, len(model.layers)):
                z[l] = (
                    np.matmul(model.layers[l].weights, a[l - 1]) + model.layers[l].bias
                )
                a[l] = model.layers[l].activation.f(z[l])

            if model.regularizer:
                weights = [layer.weights for layer in model.layers[1:]]
                regularization = model.regularizer(n, weights)
            else:
                regularization = 0

            losses[bid] = model.loss.f(y, a[-1]) + regularization

            # Ouput error
            delta = [None] * len(model.layers)

            if (
                model.loss is CrossEntropy
                and model.layers[-1].activation is Sigmoid
            ):
                delta[-1] = a[-1] - y
            elif (
                model.loss is LogLikelihood
                and model.layers[-1].activation is Softmax
            ):
                #j = np.argmax(y, axis=0)
                #delta[-1] = (
                #    model.loss.df_da(y, a[-1])
                #    * model.layers[-1].activation.df(z[-1], j)
                #)
                delta[-1] = a[-1] - y
            else:
                delta[-1] = (
                    model.loss.df_da(y, a[-1]) * model.layers[-1].activation.df(z[-1])
                )

            # Backpropagate
            for l in range(len(model.layers) - 2, 0, -1):
                delta[l] = (
                    np.matmul(model.layers[l + 1].weights.T, delta[l + 1])
                    * model.layers[l].activation.df(z[l])
                )

            # Gradient Descent
            m = x.shape[-1]
            if model.regularizer:
                weight_scale_factor = model.regularizer.weight_scale_factor(self.lr, n) 
            else:
                weight_scale_factor = 1

            for l in range(1, len(model.layers)):
                model.layers[l].weights = ( 
                    weight_scale_factor 
                    * model.layers[l].weights 
                    - self.lr / m * np.matmul(delta[l], a[l - 1].T)
                )

                model.layers[l].bias -= self.lr * np.mean(delta[l], axis=-1, keepdims=1)

            p_batches.set_description("Batches")
        return np.mean(losses)
