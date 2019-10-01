from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from .visualizer import Visualizer


class Model:
    """The Neural Net."""

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def compile(self, optimizer, loss, n_classes, regularizer):
        self.optimizer = optimizer
        self.loss = loss
        self.n_classes = n_classes
        self.regularizer = regularizer

        layers = deque()
        x = self.outputs
        while 1:
            if not hasattr(x, "prior_layer"):
                break
            x.init_bias()
            x.init_weights()

            layers.appendleft(x)
            x = x.prior_layer
        layers.appendleft(self.inputs)
        self.layers = list(layers)

    def fit(self, x, y, batch_size=None, n_epochs=1, val_data=None):
        assert len(x) == len(y)
        n = len(x)

        vis = Visualizer(n_epochs)

        y_onehot = np.eye(self.n_classes)[y]

        if not batch_size:
            batch_size = len(x)

        for e in range(n_epochs):
            print("Epoch {}:".format(e))

            loss = self.optimizer.optimize(self, x, y_onehot, batch_size)
            accu = self.evaluate(x, y)
            print("Loss: {}\tAccu: {}".format(loss, accu), end="\t")

            val_loss, val_accu = None, None
            if val_data:
                x_test, y_test = val_data

                y_test_onehot = np.eye(self.n_classes)[y_test]
                a_test = self.predict(x_test)
                if self.regularizer:
                    weights = [layer.weights for layer in self.layers[1:]]
                    regularization = self.regularizer(n, weights)
                else:
                    regularization = 0

                val_loss = self.loss.f(y_test_onehot.T, a_test.T) + regularization

                val_accu = self.evaluate(x_test, y_test)
                print("Val_loss: {}\tVal_accu: {}".format(val_loss, val_accu))
            print()

            vis.update_loss(loss, val_loss)
            vis.update_accu(accu, val_accu)
            plt.pause(0.05)
        plt.show()

    def predict(self, x):
        a = x.T
        for l in range(1, len(self.layers)):
            a = self.layers[l].activation.f(
                np.matmul(self.layers[l].weights, a) + self.layers[l].bias
            )
        return a.T

    def evaluate(self, x, y):
        predicts = self.predict(x).argmax(axis=-1)
        return len(np.where(y == predicts)[0]) / len(y)
