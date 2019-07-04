from functools import partial

import numpy as np

from .activations import Softmax
from .config import epsilon
from .utils.generic_utils import get_from_module


class Loss:
    """
    
    """
    @staticmethod
    def f(y, y_pred):
        raise NotImplementedError

    @staticmethod
    def df(y, y_pred):
        raise NotImplementedError


class MeanSquaredError(Loss):
    """
    
    """
    @staticmethod
    def f(y, y_pred):
        return np.mean(0.5 * np.sum(np.square(y - y_pred), axis=-1, keepdims=True))

    @staticmethod
    def df(y, y_pred):
        return (y_pred - y) / y.shape[0]


class MeanAbsoluteError(Loss):
    """

    """
    @staticmethod
    def f(y, y_pred):
        return np.mean(np.abs(y - y_pred), axis=-1, keepdims=True)

    @staticmethod
    def df(y, y_pred):
        return np.where(y < y_pred, 1, -1) / y.shape[0]


class Hinge(Loss):
    @staticmethod
    def f(y, y_pred):
        return np.mean(np.maximum(1.0 - y * y_pred, 0.0), axis=-1, keepdims=True)

    @staticmethod
    def df(y, y_pred):
        raise NotImplementedError


class KLDivergence(Loss):
    """
    
    """
    @staticmethod
    def f(y, y_pred):
        y = np.clip(y, epsilon, 1)
        y_pred = np.clip(y_pred, epsilon, 1)
        return np.sum(y * np.log(y / y_pred), axis=-1, keepdims=True)

    @staticmethod
    def df(y, y_pred):
        pass


class BinaryCrossentropy(Loss):
    """
    Binary crossentropy is just a special case of categorical crossentropy with 2 classes
    Reference: See the reference of CategoricalCrossentropy
    """
    @staticmethod
    def f(y, y_pred):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return np.mean(- y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))

    @staticmethod
    def df(y, y_pred):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (- (y / y_pred) + (1 - y) / (1 - y_pred)) / y.shape[0]


class CategoricalCrossentropy(Loss):
    """
    Reference: https://en.wikipedia.org/wiki/Cross_entropy
    """
    @staticmethod
    def f(y, y_pred):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return np.mean(-np.sum(y * np.log(y_pred), axis=-1, keepdims=True))

    @staticmethod
    def df(y, y_pred):
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return - (y / y_pred) / y.shape[0]


class SoftmaxCrossentropy(Loss):
    @staticmethod
    def f(y, y_pred):
        y_pred = np.clip(Softmax.f(y_pred), epsilon, 1 - epsilon)
        return np.mean(-np.sum(y * np.log(y_pred), axis=-1, keepdims=True))

    @staticmethod
    def df(y, y_pred):
        return (Softmax.f(y_pred) - y) / y.shape[0]


class CosineProximity(Loss):
    @staticmethod
    def f(y, y_pred):
        y = y / np.linalg.norm(y, axis=-1)
        y_pred = y_pred / np.linalg.norm(y, axis=-1)
        return -np.sum(y * y_pred, axis=-1)

    @staticmethod
    def df(y, y_pred):
        raise NotImplementedError


class Poisson(Loss):
    @staticmethod
    def f(y, y_pred):
        return np.mean(y_pred - y * np.log(y_pred + epsilon, axis=-1))

    @staticmethod
    def df(y, y_pred):
        raise NotImplementedError


# Aliases:
mse = MeanSquaredError
mean_squared_error = MeanSquaredError
mae = MeanAbsoluteError
mean_absolute_error = MeanAbsoluteError
hinge = Hinge
kl_divergence = KLDivergence
categorical_crossentropy = CategoricalCrossentropy
binary_crossentropy = BinaryCrossentropy
softmax_crossentropy = SoftmaxCrossentropy
cosine_proximity = CosineProximity
poisson = Poisson


LOSSES = {
    "mse": MeanSquaredError,
    "mean_squared_error": MeanSquaredError,
    "mae": MeanAbsoluteError,
    "mean_absolute_error": MeanAbsoluteError,
    "hinge": Hinge,
    "kl_divergence": KLDivergence,
    "categorical_crossentropy": CategoricalCrossentropy,
    "binary_crossentropy": BinaryCrossentropy,
    "softmax_crossentropy": SoftmaxCrossentropy,
    "cosine_proximity": CosineProximity,
    "poisson": Poisson
}

get_loss = partial(get_from_module, LOSSES, "loss")
