"""Most commonly used weight initializers along with some initializer-related utilities."""
import math
from functools import reduce, partial

import numpy as np

from .utils.generic_utils import get_from_module


class Initializer:
    """Initializer base class - all initializers should inherit from it."""
    def __call__(self, shape, dtype="float32"):
        raise NotImplementedError


class Zeros(Initializer):
    """Generates tensors filled with zeros."""
    def __call__(self, shape, dtype="float32"):
        return np.zeros(shape=shape, dtype=dtype)


class Ones(Initializer):
    """Generates tensors filled with ones."""
    def __call__(self, shape, dtype="float32"):
        return np.ones(shape=shape, dtype=dtype)


class Constant(Initializer):
    """Generates tensors filled with specified constant values.

    Attributes:
        value: float, the value to fill the tensor with
    """
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self, shape, dtype='float32'):
        return np.full(shape, self.value, dtype=dtype)


class Uniform(Initializer):
    """Generates tensors with values sampled from uniform distribution.

    Attributes:
        low: float, lower bound of the range of possible random values
        high: float, upper bound of the range of possible random values
        seed: int, used to seed the random number generator
    """
    def __init__(self, low=-1.0, high=1.0, seed=None):
        self.low = low
        self.high = high
        self.seed = seed

    def __call__(self, shape, dtype="float32"):
        return np.random.uniform(self.low, self.high, size=shape).astype(dtype)


class Normal(Initializer):
    """Generates tensors with values sampled from normal distribution.

    Attributes:
        mean: float, mean of the distribution, defaults to 0
        stddev: float, standard deviation, defaults to 0.05
        seed: int, used to seed the random number generator
    """
    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape, dtype="float32"):
        return np.random.normal(self.mean, self.stddev, size=shape).astype(dtype)


class TruncatedNormal(Initializer):
    """Generates tensors with values sampled from truncated normal distribution.

    Values more than two standard deviations from the mean are
    discarded and re-drawn from the normal distribution.

    Attributes:
        mean: float, mean of the distribution, defaults to 0
        stddev: float, standard deviation, defaults to 0.05
        seed: int, used to seed the random number generator
    """
    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape, dtype="float32"):
        x = np.random.normal(self.mean, self.stddev, shape)
        y = x[abs(x) > 2*self.stddev]
        while y.size:
            x[abs(x) > 2*self.stddev] = np.random.normal(self.mean, self.stddev, y.shape)
            y = x[abs(x) > 2*self.stddev]
        return x


class VarianceScaling(Initializer):
    """Generates tensors with values from either normal or uniform
    distribution with their scale adapted to the shape of weights.

    When `distribution="normal"`, samples are drawn from a truncated normal
    distribution with zero mean and `stddev = sqrt(gain / n)` where n is:
        - number of input units in the weight tensor, if mode = "fan_in"
        - number of output units, if mode = "fan_out"
        - average of the numbers of input and output units, if mode = "fan_avg"

    When `distribution="uniform"`,
    samples are drawn from a uniform distribution
    within [-limit, limit], where `limit = sqrt(3 * gain / n)`.

    Attributes:
        gain: float, scaling factor
        mode: str, one of {"fan_in", "fan_out", "fan_avg"}
        distribution: str, can be either "normal" or "uniform"
        seed: int, used to seed the random number generator
    """
    def __init__(self, gain=1.0,
                 mode="fan_in",
                 distribution="normal",
                 seed=None):
        if gain < 0.0:
            raise ValueError("`gain` cannot be negative")

        if mode.lower() not in {"fan_in", "fan_avg", "fan_out"}:
            raise ValueError("Invalid `mode` argument: expected "
                             "one of: 'fan_in', 'fan_avg', "
                             "'fan_out'; got: " + mode)

        if distribution.lower() not in {"normal", "uniform"}:
            raise ValueError("Invalid `distribution` argument:"
                             "expected either 'normal' or 'uniform'; "
                             "got: " + distribution)

        self.gain = gain
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def __call__(self, shape, dtype="float32"):
        gain = self.gain
        if self.mode == "fan_in":
            gain /= max(1.0, _fan_in(shape))
        elif self.mode == "fan_out":
            gain /= max(1.0, _fan_out(shape))
        else:
            gain /= max(1.0, (_fan_in(shape) + _fan_out(shape)) / 2)

        if self.distribution == "normal":
            stddev = np.sqrt(gain) / 0.87962566103423978
            return TruncatedNormal(mean=0, stddev=stddev, seed=self.seed)(shape, dtype=dtype)
        else:
            limit = np.sqrt(3.0 * gain)
            return np.random.uniform(low=-limit, high=limit,
                                     size=shape).astype(dtype)


class Orthogonal(Initializer):
    """Generates random orthogonal matrix.

    Attributes:
        gain: float, scaling factor
        seed: int, used to seed the random number generator
    """
    def __init__(self, gain=1.0, seed=None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape, dtype='float32'):
        nrows = reduce(lambda x, y: x*y, shape[:-1])
        ncols = shape[-1]
        flat_shape = (nrows, ncols)
        np.random.seed(self.seed)
        u, s, v = np.linalg.svd(np.random.normal(0.0, 1.0, flat_shape),
                                full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return self.gain * q[:shape[0], :shape[1]].astype(dtype)


class Identity(Initializer):
    """Generates identity matrix.

    Can only be duse for 2D matrices.

    Attributes:
        gain: float, scaling factor
    """
    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, shape, dtype='float32'):
        if len(shape) > 2 or shape[0] != shape[1]:
            raise ValueError("Identity matrix initializer can only be used for square matrices!")
        return self.gain * np.identity(shape[0], dtype=dtype)


def lecun_uniform(seed=None):
    """"LeCun uniform initializer.

    Generates a tensor with values according to the method described
    in "Efficient BackProp" - LeCun, Y. et. al. (1998),, using a
    uniform distribution.

    Samples are drawn from a uniform distribution within [-limit, limit]
    where `limit = sqrt(3 / fan_in)
    where `fan_in` is the number of input units in the weight tensor

    Args:
        seed: int, used to seed the random number generator

    Returns:
        instance of VarianceScaling initializer

    References:
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    """
    return VarianceScaling(gain=1.0, mode='fan_in',
                           distribution='uniform', seed=seed)


def lecun_normal(seed=None):
    """

    Generates a tensor with values according to the method described
    in "Efficient BackProp" - LeCun, Y. et. al. (1998), using a
    normal distribution.

    Samples are drawn from a truncated normal distribution with 0
    mean and `stddev = sqrt(1 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor

    Args:
        seed: int, used to seed the random number generator

    Returns:
        instance of VarianceScaling initializer

    References:
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    """
    return VarianceScaling(gain=1.0, mode='fan_in',
                           distribution='normal', seed=seed)


def xavier_uniform(seed=None):
    """Xavier (Glorot) uniform initializer.

    Generates a tensor with values according to the method described
    in "Understanding the difficulty of training deep feedforward
    neural networks" - Glorot, X. & Bengio, Y. (2010), using a
    uniform distribution.

    Samples are drawn from a uniform distribution within [-limit, limit]
    where `limit = sqrt(6 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.

    Args:
        seed: int, used to seed the random number generator

    Returns:
        instance of VarianceScaling initializer

    References:
        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    """
    return VarianceScaling(gain=1.0, mode='fan_avg',
                           distribution='uniform', seed=seed)


def xavier_normal(seed=None):
    """Xavier (Glorot) normal initializer.

    Generates a tensor with values according to the method described
    in "Understanding the difficulty of training deep feedforward
    neural networks" - Glorot, X. & Bengio, Y. (2010), using a
    normal distribution.

    Samples are drawn from a truncated normal distribution with 0
    mean and `stddev = sqrt(2 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.

    Args:
        seed: int, used to seed the random number generator

    Returns:
        instance of VarianceScaling initializer

    References:
        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    """
    return VarianceScaling(gain=1.0, mode='fan_avg',
                           distribution='normal', seed=seed)


def he_uniform(seed=None):
    """He (Kaiming) uniform initializer.

    Generates a tensor with values according to the method described
    in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015),
    using a uniform distribution.

    Samples are drawn from a uniform distribution within [-limit, limit]
    where `limit = sqrt(6 / fan_in)
    where `fan_in` is the number of input units in the weight tensor

    Args:
        seed: int, used to seed the random number generator

    Returns:
        instance of VarianceScaling initializer

    References:
        https://arxiv.org/pdf/1502.01852.pdf
    """
    return VarianceScaling(gain=2.0, mode="fan_in",
                           distribution="uniform", seed=seed)


def he_normal(seed=None):
    """He (Kaiming) normal initializer.

    Generates a tensor with values according to the method described
    in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015),
    using a uniform distribution.

    Samples are drawn from a truncated normal distribution with 0
    mean and `stddev = sqrt(2 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor


    Args:
        seed: int, used to seed the random number generator

    Returns:
        instance of VarianceScaling initializer

    References:
        https://arxiv.org/pdf/1502.01852.pdf
    """
    return VarianceScaling(gain=2.0, mode="fan_in",
                           distribution="normal", seed=seed)


# Utility functions
def calculate_gain(nonlinearity, param=None):
    """Calculates the recommended gain value for the provided nonlinearity function.

    The values are as follows:
    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky ReLU        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    ================= ====================================================

    Note:
        The function was taken from PyTorch with slight modifications.

    Args:
        nonlinearity: str, nonlinearity identifier
        param: integer or a float, optional parameter for the non-linear function

    Returns:
        float, recommended gain for the specified nonlinearity
    """
    linear_fns = [
        "linear", "conv1d", "conv2d", "conv3d",
        "conv_transpose1d", "conv_transpose2d",
        "conv_transpose3d"
    ]

    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1.0
    elif nonlinearity == "tanh":
        return 5.0 / 3.0
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and (isinstance(param, int) or isinstance(param, float)):
            negative_slope = param
        else:
            raise TypeError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def _fan_in(shape):
    if len(shape) == 2:
        return shape[0]
    elif len(shape) in {3, 4, 5}:
        return shape[1] * np.prod(shape[2:])
    else:
        return np.sqrt(np.prod(shape))


def _fan_out(shape):
    if len(shape) == 2:
        return shape[1]
    elif len(shape) in {3, 4, 5}:
        return shape[0] * np.prod(shape[2:])
    else:
        return np.sqrt(np.prod(shape))


# Aliases:
zeros = Zeros
ones = Ones
constant = Constant
normal = Normal
uniform = Uniform
truncated_normal = truncatedNormal = TruncatedNormal
variance_scaling = varianceScaling = VarianceScaling
identity = Identity
orthogonal = Orthogonal
Lecun_normal = lecunNormal = LecunNormal = lecun_normal
Lecun_uniform = lecunUniform = LecunUniform = lecun_uniform
Xavier_normal = xavierNormal = XavierNormal = xavier_normal
Xavier_uniform = xavierUniform = XavierUniform = xavier_uniform
kaiming_normal = He_normal = heNormal = HeNormal = he_normal
kaiming_uniform = He_uniform = heUniform = HeUniform = he_uniform


INITIALIZERS = {
    "zeros": Zeros(),
    "ones": Ones(),
    "constant": Constant(),
    "uniform": Uniform(),
    "normal": Normal(),
    "turncnorm": TruncatedNormal(),
    "truncated_normal": TruncatedNormal(),
    "orthogonal": Orthogonal(),
    "identity": Identity(),
    "variance_scaling": VarianceScaling(),
    "xavier_uniform": xavier_uniform(),
    "xavier_normal": xavier_normal(),
    "he_uniform": he_uniform(),
    "he_normal": he_normal(),
    "kaiming_uniform": he_uniform(),
    "kaiming_normal": he_normal(),
    "lecun_uniform": lecun_uniform(),
    "lecun_normal": lecun_normal()
}


get_initializer = partial(get_from_module, INITIALIZERS, "initializer")
