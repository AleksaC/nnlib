"""Core nnlib layers"""
import numpy as np

from .. import initializers
from ..activations import ActivationFunction, get_activation


class Layer:
    """Layer abstract base class
    """
    def __init__(self, trainable):
        self._trainable = trainable
        self.initialized = False
        self.uuid = _generate_uuid(self)

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __or__(self, other):
        raise NotImplementedError

    def __str__(self, *args, **kwargs):
        if self.initialized:
            rep = "{layer_uuid}, shape: {shape}, output shape: {output_shape}, number of parameters: {nb_params:,}"\
                .format(layer_uuid=self.uuid, shape=(self.shape()), output_shape=self.output_shape(),
                        nb_params=self.number_of_parameters())
        else:
            rep = "{layer_uuid}(uninitialized)".format(layer_uuid=self.uuid)

        return rep

    def initialize(self, input_shape):
        raise NotImplementedError

    def output_shape(self):
        raise NotImplementedError

    def number_of_parameters(self):
        raise NotImplementedError

    def shape(self):
        raise NotImplementedError

    def forward(self, x):
        pass

    def forward_inference(self, x):
        return self.forward(x)

    def backward(self, delta):
            pass


class FullyConnected(Layer):
    """

    """
    def __init__(self, number_of_units, input_shape=None,
                 weight_initializer="he_uniform", bias_initializer="zeros",
                 use_bias=True, trainable=True, activation=None):
        super().__init__(trainable)
        self.x = None
        self.z = None
        self.W = None
        self.b = None
        self.number_of_units = number_of_units
        self.input_shape = input_shape

        if isinstance(weight_initializer, str):
            self.weight_initializer = initializers.get_initializer(weight_initializer)
        elif isinstance(weight_initializer, initializers.Initializer):
            self.weight_initializer = weight_initializer
        else:
            raise TypeError("Weight initializer should be specified either by passing "
                            "a subclass of abstract class `initializers.Initializer` "
                            "or a string identifier of the initializer", weight_initializer)

        if isinstance(bias_initializer, str):
            self.bias_initializer = initializers.get_initializer(bias_initializer)
        elif isinstance(bias_initializer, initializers.Initializer):
            self.bias_initializer = bias_initializer
        else:
            raise TypeError("Bias initializer should be specified either by passing "
                            "a subclass of abstract class `initializers.Initializer` "
                            "or a string identifier of the initializer", bias_initializer)

        if activation is None:
            self.activation = None
        elif isinstance(activation, str):
            self.activation = get_activation(activation)
        elif isinstance(activation, ActivationFunction):
            self.activation = activation
        else:
            raise TypeError("Activation function of the layer should be specified"
                            "either by passing a subclass of abstract class `activations.ActivationFunction` "
                            "or by passing a string identifier of the activation function!", activation)

        self.use_bias = use_bias
        if not use_bias:
            self._forward  = self._forward_without_bias
            self._backward = self._backward_without_bias

        self.forward = self._forward if activation is None else self._forward_with_activation
        self.backward = self._backward if activation is None else self._backward_with_activation

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __or__(self, other):
        raise NotImplementedError

    def initialize(self, input_shape):
        self.initialized = True
        self.input_shape = input_shape
        self.W = self.weight_initializer(shape=(input_shape[-1], self.number_of_units))
        if self.use_bias:
            self.b = self.bias_initializer(shape=(self.number_of_units,))

    def output_shape(self):
        return self.input_shape[:-1] + (self.number_of_units,)

    def number_of_parameters(self):
        num = self.W.size
        if self.use_bias:
            num += self.b.size
        return num

    def shape(self):
        return self.input_shape[-1], self.number_of_units

    def _forward_without_bias(self, x):
        self.x = x
        self.z = x.dot(self.W)
        return self.z

    def _backward_without_bias(self, delta):
        return delta.dot(self.W.T), (self.x.T.dot(delta), 0)

    def _forward(self, x):
        self.x = x
        self.z = x.dot(self.W) + self.b
        return self.z

    def _backward(self, delta):
        dw = self.x.T.dot(delta)
        db = np.sum(delta, axis=0)
        return delta.dot(self.W.T), (dw, db)

    def _forward_with_activation(self, x):
        self._forward(x)
        return self.activation.f(self.z)

    def _backward_with_activation(self, delta):
        delta *= self.activation.df(self.z)
        return self._backward(delta)

    def update_trainable_params(self, grads):
        raise NotImplementedError


layer_num = {}


def _generate_uuid(layer):
    global layer_num
    layer_name = layer.__class__.__name__
    num = layer_num.get(layer_name, 0)
    uuid = layer_name + str(num)
    num += 1
    layer_num[layer_name] = num
    return uuid
