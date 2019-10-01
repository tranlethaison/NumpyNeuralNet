import numpy as np


class MSE:
    @staticmethod
    def f(y, a):
        return np.mean(0.5 * np.sum(np.square(y - a), axis=0))

    @staticmethod
    def df_da(y, a):
        """Return partial derivative wrt `a` (element-wise)."""
        return a - y


class CrossEntropy:
    @staticmethod
    def f(y, a):
        return np.mean(-np.sum(y * np.log(a) + (1 - y) * np.log(1 - a), axis=0))

    @staticmethod
    def df_da(y, a):
        """Return partial derivative wrt `a` (element-wise)."""
        return -(y / a + (y - 1) / (1 - a))


class LogLikelihood:
    @staticmethod
    def f(y, a):
        j = np.argmax(y, axis=0)
        losses = np.zeros(j.shape)

        for sid in range(len(losses)):
            losses[sid] = -np.log(a[j[sid], sid])
        return np.mean(losses)

    @staticmethod
    def df_da(y, a):
        """Return partial derivative wrt `a` (element-wise)."""
        j = np.argmax(y, axis=0)
        r = np.zeros(a.shape)

        for sid in range(r.shape[-1]):
            r[:, sid] = -1 / a[j[sid], sid]
        return r
