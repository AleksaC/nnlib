# Reference: https://en.wikipedia.org/wiki/Activation_function
from functools import partial

import numpy as np

from .utils.generic_utils import get_from_module


class ActivationFunction:
    """Activation function base class"""
    @staticmethod
    def f(x):
        raise NotImplementedError

    @staticmethod
    def df(x):
        raise NotImplementedError


class Linear(ActivationFunction):
    @staticmethod
    def f(x):
        return x

    @staticmethod
    def df(x):
        return 1


class Sigmoid(ActivationFunction):
    """
    Reference: https://en.wikipedia.org/wiki/Sigmoid_function
    """
    @staticmethod
    def f(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def df(x):
        f = Sigmoid.f(x)
        return f * (1 - f)


class Softplus(ActivationFunction):
    """

    """
    @staticmethod
    def f(x):
        return np.log(1 + np.exp(x))

    @staticmethod
    def df(x):
        return Sigmoid.f(x)


class Softsign(ActivationFunction):
    """

    """
    @staticmethod
    def f(x):
        return x / (1 + np.abs(x))

    @staticmethod
    def df(x):
        return 1 / np.square(1 + np.abs(x))


class Tanh(ActivationFunction):
    """
    Reference:
    """
    @staticmethod
    def f(x):
        e_2x = np.exp(-2*x)
        return (1 - e_2x) / (1 + e_2x)

    @staticmethod
    def df(x):
        return 1 - np.square(Tanh.f(x))


class ReLU(ActivationFunction):
    """
    Reference: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

    """
    @staticmethod
    def f(x):
        return np.where(x > 0, x, 0)

    @staticmethod
    def df(x):
        return np.where(x > 0, 1, 0)


class Softmax(ActivationFunction):
    """
    Reference: https://en.wikipedia.org/wiki/Softmax_function
    """
    @staticmethod
    def f(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    @staticmethod
    def df(x):
        grad = []
        for x_i in x:
            dx_i = np.sum(np.diag(x_i) - x_i * x_i.reshape(-1, 1), axis=-1)
            grad.append(dx_i)
        return np.array(grad)


class ELU(ActivationFunction):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def f(self, x):
        return np.where(x < 0, self.alpha * (np.exp(x)-1), x)

    def df(self, x):
        return np.where(x < 0, self.alpha * np.exp(x), 1)


class SELU(ActivationFunction):
    # Reference: Klambauer et al.(2017), https://arxiv.org/abs/1706.02515
    # Code sample: https://github.com/bioinf-jku/SNNs/blob/master/SelfNormalizingNetworks_CNN_MNIST.ipynb
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    @staticmethod
    def f(x):
        return np.multiply(SELU.scale, np.where(x < 0, SELU.alpha * (np.exp(x)-1), x))

    @staticmethod
    def df(x):
        return np.multiply(SELU.scale, np.where(x < 0, SELU.alpha * np.exp(x), 1))


ACTIVATIONS = {
    "linear": Linear,
    "tanh": Tanh,
    "sigmoid": Sigmoid,
    "softplus": Softplus,
    "softmax": Softmax,
    "relu": ReLU,
    "elu": ELU
}

get_activation = partial(get_from_module, ACTIVATIONS, "activation")
