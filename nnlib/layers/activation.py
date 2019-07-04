"""nnlib activation layers

"""
import numpy as np

from .core import Layer
from ..activations import get_activation
from ..activations import ActivationFunction
from ..initializers import get_initializer
from ..initializers import Initializer


class _Activation(Layer):
    """Activation layer abstract base class - all activation
    layers should inherit from it.
    """
    def __init__(self):
        super().__init__(trainable=False)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __or__(self, other):
        raise NotImplementedError

    @Layer.trainable.setter
    def trainable(self, value):
        raise AttributeError("Cannot change trainable property of an activation layer")

    def f(self, x):
        raise NotImplementedError

    def df(self, x):
        raise NotImplementedError

    def forward(self, x):
        self.x = x
        return self.f(x)

    def backward(self, delta):
        return delta * self.df(self.x), None

    def initialize(self, input_shape):
        self.initialized = True
        self.input_shape = input_shape

    def output_shape(self):
        return self.input_shape

    def number_of_parameters(self):
        return 0

    def shape(self):
        return None


class Activation(_Activation):
    """Activation layer

    Attributes:
        activation: act
    """
    def __init__(self, activation):
        super().__init__()

        if isinstance(activation, str):
            activation = get_activation(activation)
        elif not issubclass(activation, ActivationFunction):
            raise TypeError("Activation ")

        self.f = activation.f
        self.df = activation.df

    def __call__(self, *args, **kwargs):
        pass

    def __or__(self, other):
        pass

    def f(self, x):
        pass

    def df(self, x):
        pass


class PReLU(_Activation):
    """Parametric ReLU activation layer

    Attributes:
        alpha_initializer: str or Initializer, values initial values
        of alpha tensor

    References:
        https://arxiv.org/pdf/1502.01852.pdf
    """
    def __init__(self, alpha_initializer="constant"):
        super().__init__()

        self.trainable = True
        self.alpha = None

        if isinstance(alpha_initializer, str):
            self.alpha_initializer = get_initializer(alpha_initializer)
            if alpha_initializer == "constant":
                self.alpha_initializer.value = 0.25
        elif isinstance(alpha_initializer, Initializer):
            self.alpha_initializer = alpha_initializer
        else:
            raise TypeError("Alpha initializer should be specified either by passing a class "
                            "that inherits from abstract class `initializers.Initializer` "
                            "or a string identifier of the initializer", alpha_initializer)

    @_Activation.trainable.setter
    def trainable(self, value):
        self._trainable = value

    def __call__(self, *args, **kwargs):
        pass

    def __or__(self, other):
        pass

    def f(self, x):
        return np.where(x > 0.0, x, self.alpha * x)

    def df(self, x):
        return np.where(self.x > 0.0, 0.0, self.x)

    def initialize(self, input_shape):
        super().initialize(input_shape)
        self.alpha = self.alpha_initializer(input_shape)

    def forward(self, x):
        self.x = x
        return self.f(x)

    def backward(self, delta):
        dalpha = self.df(self.x) * delta
        return delta * self.df(self.x), dalpha
