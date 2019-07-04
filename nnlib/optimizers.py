"""Implementations of most commonly used optimizers

This module contains implementations of optimization algorithms
most commonly used in deep learning.

Reference: https://arxiv.org/pdf/1609.04747.pdf
"""

from functools import partial

import numpy as np

from .utils.generic_utils import get_from_module
from . import config


class Optimizer:
    """Optimizer abstract base class.

    This class implements
    """
    def __init__(self, lr, decay=0.0, clip_norm=0.0, clip_value=0.0):
        if clip_norm < 0:
            raise ValueError("Clip norm should be positive!")
        if clip_value < 0:
            raise ValueError("Clip value should be positive!")

        self.lr = lr
        self.decay = decay
        self.iterations = 0
        self.clip_norm = clip_norm
        self.clip_value = clip_value

        self.model = None

        if clip_value != 0 or clip_norm != 0:
            self.compute_grads = self._compute_grads_with_clipping
        else:
            self.compute_grads = self._compute_grads

    def update(self, params, grads):
        raise NotImplementedError

    def compute_grads(self, x, y):
        pass

    def _compute_grads(self, x, y):
        self.x = x
        self.y = y

        y_pred = self.model._forward(x)
        grads  = self.model._backward(self.model.loss.df(y, y_pred))

        return grads, y_pred

    def _compute_grads_with_clipping(self, x, y):
        grads, y_pred = self._compute_grads(x, y)

        if self.clip_norm != 0:
            self.clip_by_norm(grads)
        if self.clip_value != 0:
            self.clip_by_value(grads)

        return grads, y_pred

    def clip_by_value(self, grads):
        for layer_uuid, grad in grads.items():
            grads[layer_uuid] = np.clip(grad[0], -self.clip_value, self.clip_value), \
                                np.clip(grad[1], -self.clip_value, self.clip_value)

    def clip_by_norm(self, grads):
        for layer_uuid, grad in grads.items():
            norm = np.linalg.norm(grad[0])
            if norm > self.clip_norm:
                grads[layer_uuid][0] = grad[0] * self.clip_norm / norm
            norm = np.linalg.norm(grad[1])
            if norm > self.clip_norm:
                grads[layer_uuid][1] = grad[1] * self.clip_norm / norm


class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9, nesterov=False,
                 decay=0.0, clip_norm=0.0, clip_value=0.0):
        super().__init__(lr, decay, clip_norm, clip_value)

        self.momentum = momentum
        self.prev_grads = {}

        self._nesterov = nesterov
        self.update = self._nesterov_update if nesterov else self._update

    @property
    def nesterov(self):
        return self._nesterov

    @nesterov.setter
    def nesterov(self, value):
        self._nesterov = value
        self.update = self._nesterov_update if self._nesterov else self._update

    def update(self, params, grads):
        pass

    def _update(self, params, grads):
        self.iterations += 1

        lr = self.lr
        if self.decay > 0:
            lr = lr * (1.0 / (1.0 + self.decay * self.iterations))

        for layer_uuid, grad in grads.items():
            dw = lr * grad[0] + self.momentum * self.prev_grads.get(layer_uuid, (0, 0))[0]
            db = lr * grad[1] + self.momentum * self.prev_grads.get(layer_uuid, (0, 0))[1]
            self.prev_grads[layer_uuid] = (dw, db)

            params[layer_uuid].W -= dw
            params[layer_uuid].b -= db

    def _nesterov_update(self, params, grads):
        self.iterations += 1

        lr = self.lr
        if self.decay > 0:
            lr = lr * (1.0 / (1.0 + self.decay * self.iterations))

        for layer_uuid, grad in grads.items():
            params[layer_uuid].W -= self.momentum * self.prev_grads.get(layer_uuid, (0, 0))[0]
            params[layer_uuid].b -= self.momentum * self.prev_grads.get(layer_uuid, (0, 0))[1]

        grads = self.compute_grads(self.x, self.y)[0]

        for layer_uuid, grad in grads.items():
            dw = lr * grad[0] + self.momentum * self.prev_grads.get(layer_uuid, (0, 0))[0]
            db = lr * grad[1] + self.momentum * self.prev_grads.get(layer_uuid, (0, 0))[1]

            params[layer_uuid].W -= dw
            params[layer_uuid].b -= db

            self.prev_grads[layer_uuid] = (dw, db)


class Adagrad(Optimizer):
    def __init__(self, lr=0.01, epsilon=config.epsilon, decay=0.0,
                 clip_norm=0.0, clip_value=0.0):
        super().__init__(lr, decay, clip_norm, clip_value)

        self.epsilon = epsilon
        self.cache = {}

    def update(self, params, grads):
        self.iterations += 1

        lr = self.lr
        if self.decay > 0:
            lr = lr * (1.0 / (1.0 + self.decay * self.iterations))

        for layer_uuid, grad in grads.items():
            dw = self.cache.get(layer_uuid, (0, 0))[0] + grad[0] ** 2
            db = self.cache.get(layer_uuid, (0, 0))[1] + grad[1] ** 2

            params[layer_uuid].W -= lr * grad[0] / np.sqrt(dw + self.epsilon)
            params[layer_uuid].b -= lr * grad[1] / np.sqrt(db + self.epsilon)

            self.cache[layer_uuid] = (dw, db)


class Adadelta(Optimizer):
    def __init__(self, lr=0.01, rho=0.9, epsilon=config.epsilon,
                 decay=0.0, clip_norm=0.0, clip_value=0.0):
        super().__init__(lr, decay, clip_norm, clip_value)

        self.rho = rho
        self.epsilon = epsilon
        self.cache = {}
        self.update_cache = {}
        self.prev_grads = {}

    def update(self, params, grads):
        self.iterations += 1

        for layer_uuid, grad in grads.items():
            dw = self.rho * self.cache.get(layer_uuid, (0, 0))[0] + \
                 (1 - self.rho) * grad[0] ** 2
            db = self.rho * self.cache.get(layer_uuid, (0, 0))[1] + \
                 (1 - self.rho) * grad[1] ** 2
            self.cache[layer_uuid] = (dw, db)

            upw = self.rho * self.update_cache.get(layer_uuid, (0, 0))[0] + \
                  (1 - self.rho) * self.prev_grads.get(layer_uuid, (0, 0))[0] ** 2
            upb = self.rho * self.update_cache.get(layer_uuid, (0, 0))[1] + \
                  (1 - self.rho) * self.prev_grads.get(layer_uuid, (0, 0))[1] ** 2
            self.update_cache[layer_uuid] = (upw, upb)

            dw = np.sqrt((upw + self.epsilon) / (dw + self.epsilon)) * grad[0]
            db = np.sqrt((upb + self.epsilon) / (db + self.epsilon)) * grad[1]

            params[layer_uuid].W -= dw
            params[layer_uuid].b -= db

            self.prev_grads[layer_uuid] = (dw, db)


class RMSProp(Optimizer):
    def __init__(self, lr=0.01, decay_rate=0.9, epsilon=config.epsilon,
                 decay=0.0, clip_norm=0.0, clip_value=0.0):
        super().__init__(lr, decay, clip_norm, clip_value)

        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}

    def update(self, params, grads):
        self.iterations += 1

        lr = self.lr
        if self.decay > 0:
            lr = lr * (1.0 / (1.0 + self.decay * self.iterations))

        for layer_uuid, grad in grads.items():
            dw = self.decay_rate * self.cache.get(layer_uuid, (0, 0))[0] + \
                 (1 - self.decay_rate) * grad[0] ** 2
            db = self.decay_rate * self.cache.get(layer_uuid, (0, 0))[1] + \
                 (1 - self.decay_rate) * grad[1] ** 2

            params[layer_uuid].W -= lr * grad[0] / np.sqrt(dw + self.epsilon)
            params[layer_uuid].b -= lr * grad[1] / np.sqrt(db + self.epsilon)

            self.cache[layer_uuid] = (dw, db)


class Adam(Optimizer):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=config.epsilon,
                 amsgrad=False, decay=0.0, clip_norm=0.0, clip_value=0.0):
        super().__init__(lr, decay, clip_norm, clip_value)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.m = {}
        self.v = {}

        if amsgrad:
            self.vt = {}
            self.update = self._amsgrad_update
        else:
            self.update = self._update

    def update(self, params, grads):
        pass

    def _update(self, params, grads):
        self.iterations += 1

        lr = self.lr
        if self.decay > 0:
            lr = lr * (1.0 / (1.0 + self.decay * self.iterations))

        for layer_uuid, grad in grads.items():
            mw = self.beta1 * self.m.get(layer_uuid, (0, 0))[0] + (1 - self.beta1) * grad[0]
            mb = self.beta1 * self.m.get(layer_uuid, (0, 0))[1] + (1 - self.beta1) * grad[1]
            self.m[layer_uuid] = (mw, mb)

            vw = self.beta2 * self.v.get(layer_uuid, (0, 0))[0] + (1 - self.beta2) * (grad[0] ** 2)
            vb = self.beta2 * self.v.get(layer_uuid, (0, 0))[1] + (1 - self.beta2) * (grad[1] ** 2)
            self.v[layer_uuid] = (vw, vb)

            mtw = mw / (1 - self.beta1 ** self.iterations)
            mtb = mb / (1 - self.beta1 ** self.iterations)

            vtw = vw / (1 - self.beta2 ** self.iterations)
            vtb = vb / (1 - self.beta2 ** self.iterations)

            params[layer_uuid].W -= lr * mtw / (np.sqrt(vtw) + self.epsilon)
            params[layer_uuid].b -= lr * mtb / (np.sqrt(vtb) + self.epsilon)

    def _amsgrad_update(self, params, grads):
        self.iterations += 1

        lr = self.lr
        if self.decay > 0:
            lr = lr * (1.0 / (1.0 + self.decay * self.iterations))

        for layer_uuid, grad in grads.items():
            mw = self.beta1 * self.m.get(layer_uuid, (0, 0))[0] + (1 - self.beta1) * grad[0]
            mb = self.beta1 * self.m.get(layer_uuid, (0, 0))[1] + (1 - self.beta1) * grad[1]
            self.m[layer_uuid] = (mw, mb)

            vw = self.beta2 * self.v.get(layer_uuid, (0, 0))[0] + (1 - self.beta2) * (grad[0] ** 2)
            vb = self.beta2 * self.v.get(layer_uuid, (0, 0))[1] + (1 - self.beta2) * (grad[1] ** 2)
            self.v[layer_uuid] = (vw, vb)

            vtw = np.maximum(self.vt.get(layer_uuid, np.zeros((1,) + vw.shape))[0], vw)
            vtb = np.maximum(self.vt.get(layer_uuid, np.zeros((2,) + vb.shape))[1], vb)
            self.vt[layer_uuid] = (vtw, vtb)

            params[layer_uuid].W -= lr * mw / (np.sqrt(vtw) + self.epsilon)
            params[layer_uuid].b -= lr * mb / (np.sqrt(vtb) + self.epsilon)


# Aliases
sgd = SGD()
adagrad = Adagrad()
adadelta = Adadelta()
rmsprop = RMSProp()
adam = Adam()
amsgrad = Adam(amsgrad=True)


OPTIMIZERS = {
    "sgd": SGD(),
    "adagrad": Adagrad(),
    "adadelta": Adadelta(),
    "rmsprop": RMSProp(),
    "adam": Adam(),
    "amsgrad": Adam(amsgrad=True)
}

get_optimizer = partial(get_from_module, OPTIMIZERS, "optimizer")
