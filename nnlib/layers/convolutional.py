"""1D and 2D convolutional layers"""
import math
import numpy as np

from .core import Layer
from .. import initializers
from ..activations import get_activation, ActivationFunction


class Conv1D(Layer):
    def __call__(self, *args, **kwargs):
        pass

    def __or__(self, other):
        pass

    def initialize(self, input_shape):
        pass

    def output_shape(self):
        pass

    def number_of_parameters(self):
        pass

    def shape(self):
        pass


class Conv2D(Layer):
    """2D (spatial) convolution layer.

    """

    def __init__(self, number_of_filters, kernel_size, stride=1, padding="valid", input_shape=None,
                 weight_initializer="he_uniform", bias_initializer="zeros",
                 use_bias=True, trainable=True, activation=None):
        super().__init__(trainable=trainable)

        self.number_of_filters = number_of_filters
        self.kernel_size = kernel_size
        self.stride = stride

        if padding.lower() not in {"same", "valid"}:
            raise ValueError("Padding can be one of the two values {{'valid','same'}}. "
                             "Got: {}".format(padding))
        self.padding = padding

        if input_shape is not None:
            if len(input_shape) < 2:
                raise ValueError("Cannot perform 2D convolution over "
                                 "one-dimensional arrays")
            if input_shape[-1] != input_shape[-2]:
                raise ValueError("2D convolution layers expect square images")
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

        self.use_bias = use_bias

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
            self._forward = self._forward_without_bias
            self._backward = self._backward_without_bias

        self.forward = self._forward if activation is None else self._forward_with_activation
        self.backward = self._backward if activation is None else self._backward_with_activation

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __or__(self, other):
        raise NotImplementedError

    def initialize(self, input_shape=None):
        if input_shape is None:
            if self.input_shape is None:
                raise ValueError()
            input_shape = self.input_shape

        if input_shape[1] != input_shape[2]:
            raise ValueError()

        if input_shape[1] < self.kernel_size:
            raise ValueError("Image size is smaller than kernel size!", self.uuid)

        if (input_shape[1] - self.kernel_size) % self.stride != 0:
            raise ValueError()

        num_channels = input_shape[0]

        self.initialized = True
        self.input_shape = input_shape

        self.W = self.weight_initializer(shape=(self.number_of_filters, num_channels * self.kernel_size ** 2))

        if self.use_bias:
            self.b = self.bias_initializer(shape=(self.number_of_filters,))

    def output_shape(self):
        if self.input_shape is None:
            raise ValueError("Cannot determine the output shape of layer {} "
                             "because its input shape hasn't been specified yet".format(self.uuid))

        width = height = (self.input_shape[1] - self.kernel_size) // self.stride + 1 \
                         if self.padding == "valid" \
                         else math.ceil((self.input_shape[1] - self.kernel_size) // self.stride + 1)

        return self.number_of_filters, width, height

    def number_of_parameters(self):
        num = self.W.size
        if self.use_bias:
            num += self.b.size
        return num

    def shape(self):
        num_channels = self.input_shape[0]
        return self.number_of_filters, num_channels, self.kernel_size, self.kernel_size

    def _forward_without_bias(self, x):
        pass

    def _backward_without_bias(self, delta):
        pass

    def _forward(self, x):
        self.x = im2col(x, self.kernel_size, self.stride)
        self.z = self.W.dot(self.x)
        return col2im(self.z)

    def _backward(self, delta):
        pass

    def _forward_with_activation(self, x):
        pass

    def _backward_with_activation(self, delta):
        pass

    def update_trainable_params(self, grads):
        pass


# Utility functions
def im2col(images, kernel_size, stride):
    """Converts a batch of images into matrices whose columns correspond
    to flattened convolution patches.

    Args:
        images: tensor, a batch of images of shape (batch_size, num_channels, height, width)
        kernel_size: int
        stride: int

    Returns:
        imcol, batch of matrices of convolution patches
    """
    num_channels = images.shape[1]
    img_size = images.shape[2]
    img_col_size = num_channels * kernel_size ** 2
    img_col_num = ((img_size - kernel_size) // stride + 1) ** 2
    batch_size = images.shape[0]
    imcol = np.empty(shape=(batch_size, img_col_size, img_col_num), dtype=images.dtype)

    k = 0
    for i in range(0, img_size - kernel_size + 1, stride):
        for j in range(0, img_size - kernel_size + 1, stride):
            imcol[:, :, k] = images[:, :, i:i + kernel_size, j:j + kernel_size].reshape(batch_size, img_col_size)
            k += 1

    return imcol


def col2im(cols):
    batch_size = cols.shape[0]
    num_filters = cols.shape[1]
    size = int(math.sqrt(cols.shape[2]))
    return cols.reshape(batch_size, num_filters, size, size)
