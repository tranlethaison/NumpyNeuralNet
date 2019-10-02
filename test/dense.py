import sys 

import numpy as np
from keras.datasets import mnist, fashion_mnist
import fire

sys.path.append(".")

from numpynn.layers import Input, Dense, Dropout
from numpynn.activations import Linear, Sigmoid, Softmax
from numpynn.initializers import RandomNormal, RandomUniform, Zeros, StandardNormal
from numpynn.models import Model
from numpynn.optimizers import SGD
from numpynn.losses import MSE, CrossEntropy, LogLikelihood
from numpynn.regularizers import L2, L1


def preprocess(x):
    return (x / 255.0).reshape(x.shape[0], -1)


def data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    n = 50000
    x_train, x_val = x_train[:n], x_train[n:]
    y_train, y_val = y_train[:n], y_train[n:]

    x_train = preprocess(x_train)
    x_val = preprocess(x_val)
    x_test = preprocess(x_test)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


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
    cfg = {"lr": 3.0, "loss": MSE}
    return model, cfg


def sigmoid_crossentropy():
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
    cfg = {"lr": 0.5, "loss": CrossEntropy}
    return model, cfg


def softmax_loglikelihood():
    inputs = Input(784)
    x = Dense(
        30,
        activation=Sigmoid,
        kernel_initializer=RandomNormal(),
        bias_initializer=Zeros(),
    )(inputs)
    outputs = Dense(
        10,
        activation=Softmax,
        kernel_initializer=RandomNormal(),
        bias_initializer=Zeros(),
    )(x)

    model = Model(inputs=inputs, outputs=outputs)
    cfg = {"lr": 0.5, "loss": LogLikelihood}
    return model, cfg


def softmax_loglikelihood_dropout():
    inputs = Input(784)
    x = Dense(
        30,
        activation=Sigmoid,
        kernel_initializer=RandomNormal(),
        bias_initializer=Zeros(),
    )(inputs)
    x = Dropout(0.5)(x)
    outputs = Dense(
        10,
        activation=Softmax,
        kernel_initializer=RandomNormal(),
        bias_initializer=Zeros(),
    )(x)

    model = Model(inputs=inputs, outputs=outputs)
    cfg = {"lr": 0.5, "loss": LogLikelihood}
    return model, cfg


def train(model, cfg):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data()

    model.compile(
        optimizer=SGD(cfg["lr"]), loss=cfg["loss"], n_classes=10, regularizer=None
    )
    model.fit(x_train, y_train, batch_size=10, n_epochs=30, val_data=(x_val, y_val))
    accuracy = model.evaluate(x_test, y_test)
    print("Accuracy:", accuracy)


def overfit_test(model, cfg):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data()
    n = 1000
    x_train, y_train = x_train[:n], y_train[:n]

    model.compile(
        #optimizer=SGD(cfg["lr"]), loss=cfg["loss"], n_classes=10, regularizer=None
        #optimizer=SGD(cfg["lr"]), loss=cfg["loss"], n_classes=10, regularizer=L2(5e-5)
        optimizer=SGD(cfg["lr"]), loss=cfg["loss"], n_classes=10, regularizer=L1(5e-5)
    )
    model.fit(x_train, y_train, batch_size=10, n_epochs=400, val_data=(x_val, y_val))
    accuracy = model.evaluate(x_test, y_test)
    print("Accuracy:", accuracy)


def main(model_nm="softmax_loglikelihood", action="train"):
    model, cfg = globals()[model_nm]()
    globals()[action](model, cfg)


if __name__ == "__main__":
    fire.Fire(main)
