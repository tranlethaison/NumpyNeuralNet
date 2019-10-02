import os

import numpy as np
from keras.datasets import mnist, fashion_mnist
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.activations import sigmoid, softmax
from tensorflow.keras.initializers import RandomNormal, Zeros
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.keras.optimizers import SGD
import fire


def preprocess(x, y):
    x = (x / 255.0).reshape(x.shape[0], -1)
    y = np.eye(10)[y]
    return x, y


def data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    n = 50000
    x_train, x_val = x_train[:n], x_train[n:]
    y_train, y_val = y_train[:n], y_train[n:]

    xs = [x_train, x_val, x_test]
    ys = [y_train, y_val, y_test]
    return list(map(preprocess, xs, ys))


def sigmoid_crossentropy():
    inputs = Input(784)
    x = Dense(
        30,
        activation=sigmoid,
        kernel_initializer=RandomNormal,
        bias_initializer=Zeros,
        kernel_regularizer=L1L2(l2=5e-5),
    )(inputs)
    outputs = Dense(
        10,
        activation=sigmoid,
        kernel_initializer=RandomNormal,
        bias_initializer=Zeros,
        kernel_regularizer=L1L2(l2=5e-5),
    )(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    cfg = {"lr": 0.5, "loss": binary_crossentropy}
    return model, cfg


def train(model, cfg):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data()

    model.compile(
        optimizer=SGD(learning_rate=cfg["lr"]), loss=cfg["loss"], metrics=["accuracy"]
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=10,
        epochs=1,
        validation_data=(x_val, y_val),
        shuffle=True,
    )
    val = model.evaluate(x_val, y_val, batch_size=len(x_val), verbose=0)
    print(val)


def main(model_nm="softmax_loglikelihood", action="train"):
    model, cfg = globals()[model_nm]()
    globals()[action](model, cfg)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    fire.Fire(main)
