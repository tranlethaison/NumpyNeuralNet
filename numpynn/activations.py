import numpy as np


class Linear:
    @staticmethod
    def f(z):
        return z

    @staticmethod
    def df(z):
        return 1


class Sigmoid:
    @staticmethod
    def f(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def df(z):
        return Sigmoid.f(z) * (1.0 - Sigmoid.f(z))


class Softmax:
    @staticmethod
    def f(z):
        e_z = np.exp(z)
        return e_z / np.sum(e_z, axis=0, keepdims=True)

    @staticmethod
    def df(z, j):
        assert z.shape[-1] == len(j)

        class_ids = np.arange(z.shape[0])
        k = [np.where(class_ids != jj) for jj in j]

        a = Softmax.f(z)
        r = np.zeros(a.shape)

        for sid in range(r.shape[-1]):
            r[j[sid], sid] = a[j[sid], sid] * (1 - a[j[sid], sid])
            r[k[sid], sid] = -a[j[sid], sid] * a[k[sid], sid]

        return r
