import numpy as np

from .core import Layer


class Dropout(Layer):
    """Dropout layer.

    Each input of the layer is multiplied by zero with the probability
    specified, which is equivalent to "dropping out" the unit in a previous
    layer whose output was zeroed out.

    Attributes:
        prob: float, probability of dropping out a unit

    # Reference:
        http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
    """
    def __init__(self, prob):
        if not isinstance(prob, float):
            raise TypeError("")

        if not 0 < prob < 1:
            raise ValueError("")

        super().__init__(trainable=False)
        self.prob = prob
        self.trainable = False

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __or__(self, other):
        raise NotImplementedError

    def initialize(self, input_shape):
        self.initialized = True
        self.input_shape = input_shape

    def output_shape(self):
        return self.input_shape

    def number_of_parameters(self):
        return 0

    def shape(self):
        return None

    def forward(self, x):
        self.x = x
        self.dropped_out_units = np.random.binomial(1, self.prob, x.shape)
        return x * self.dropped_out_units

    def forward_inference(self, x):
        return (1 - self.prob) * x

    def backward(self, delta):
        return delta * self.dropped_out_units, None


class BatchNormalization(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __or__(self, other):
        raise NotImplementedError

    def initialize(self, input_shape):
        pass

    def output_shape(self):
        pass

    def number_of_parameters(self):
        pass

    def shape(self):
        pass

    def update_trainable_params(self, grads):
        pass

    def forward(self, x):
        pass

    def backward(self, delta):
        pass
