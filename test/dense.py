import sys

import numpy as np
from keras.datasets import mnist, fashion_mnist
import fire

sys.path.append(".")

from numpynn.layers import *
from numpynn.activations import *
from numpynn.initializers import *
from numpynn.models import Model
from numpynn.optimizers import SGD
from numpynn.losses import *
from numpynn.regularizers import *


def data(scale=[0, 1]):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    n = 50000
    x_train, x_val = x_train[:n], x_train[n:]
    y_train, y_val = y_train[:n], y_train[n:]

    x_train = preprocess(x_train, scale)
    x_val = preprocess(x_val, scale)
    x_test = preprocess(x_test, scale)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def preprocess(x, scale):
    return normalize(x, scale).reshape(x.shape[0], -1)


def normalize(x, scale):
    a, b = scale
    min_, max_ = 0, 255
    return ((b - a) * (x - min_)) / (max_ - min_) + a


def sigmoid_mse():
    inputs = Input(784)
    x = Dense(
        30,
        activation=Sigmoid,
        kernel_initializer=RandomNormal(),
        bias_initializer=Zeros(),
    )(inputs)
    outputs = Dense(
        10,
        activation=Sigmoid,
        kernel_initializer=RandomNormal(),
        bias_initializer=Zeros(),
    )(x)

    model = Model(inputs=inputs, outputs=outputs)
    cfg = {
        "optimizer": SGD(lr=3.0, momentum=0.0), 
        "loss": MSE,
        "scale": [0, 1],
    }
    return model, cfg


def sigmoid_crossentropy():
    inputs = Input(784)
    x = Dense(
        30,
        activation=Sigmoid,
        kernel_initializer=RandomNormal(0, 1 / (784 ** 0.5)),
        bias_initializer=Zeros(),
        kernel_regularizer=L2(5e-5),
    )(inputs)
    outputs = Dense(
        10,
        activation=Sigmoid,
        kernel_initializer=RandomNormal(0, 1 / (30 ** 0.5)),
        bias_initializer=Zeros(),
        kernel_regularizer=L2(5e-5),
    )(x)

    model = Model(inputs=inputs, outputs=outputs)
    cfg = {
        "optimizer": SGD(lr=0.5, momentum=0.2),
        "loss": CrossEntropy,
        "scale": [0, 1],
    }
    return model, cfg


def softmax_loglikelihood():
    inputs = Input(784)
    x = Dense(
        30,
        activation=Sigmoid,
        kernel_initializer=RandomNormal(0, 1 / (784 ** 0.5)),
        bias_initializer=Zeros(),
        kernel_regularizer=L2(5e-5),
    )(inputs)
    # x = Dropout(0.5)(x)
    outputs = Dense(
        10,
        activation=Softmax,
        kernel_initializer=RandomNormal(0, 1 / (30 ** 0.5)),
        bias_initializer=Zeros(),
        kernel_regularizer=L2(5e-5),
    )(x)

    model = Model(inputs=inputs, outputs=outputs)
    cfg = {
        "optimizer": SGD(lr=0.5, momentum=0.2), 
        "loss": LogLikelihood,
        "scale": [0, 1],
    }
    return model, cfg


def relu_mse():
    inputs = Input(784)
    x = Dense(
        30,
        activation=ReLU,
        kernel_initializer=RandomNormal(0, 1 / (784 ** 0.5)),
        bias_initializer=Zeros(),
        kernel_regularizer=L2(5e-5),
    )(inputs)
    # x = Dropout(0.5)(x)
    outputs = Dense(
        10,
        activation=ReLU,
        kernel_initializer=RandomNormal(0, 1 / (30 ** 0.5)),
        bias_initializer=Zeros(),
        kernel_regularizer=L2(5e-5),
    )(x)

    model = Model(inputs=inputs, outputs=outputs)
    cfg = {
        "optimizer": SGD(lr=0.25, momentum=0.2), 
        "loss": MSE,
        "scale": [0, 1],
    }
    return model, cfg


def train(model, cfg):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data(cfg["scale"])

    model.compile(optimizer=cfg["optimizer"], loss=cfg["loss"], n_classes=10)
    model.fit(x_train, y_train, batch_size=10, n_epochs=30, val_data=(x_val, y_val))
    accuracy = model.evaluate(x_test, y_test)
    print("Accuracy:", accuracy)


def small_train(model, cfg):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data(cfg["scale"])
    n_train = 10000
    n_val = 2000
    x_train, y_train = x_train[:n_train], y_train[:n_train]
    x_val, y_val = x_val[:n_val], y_val[:n_val]

    model.compile(optimizer=cfg["optimizer"], loss=cfg["loss"], n_classes=10)
    model.fit(x_train, y_train, batch_size=10, n_epochs=30, val_data=(x_val, y_val))


def main(model_nm="softmax_loglikelihood", action="train"):
    model, cfg = globals()[model_nm]()
    globals()[action](model, cfg)


if __name__ == "__main__":
    fire.Fire(main)
