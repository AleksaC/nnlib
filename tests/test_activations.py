import pytest
import numpy as np
from numpy.testing import assert_allclose
from nnlib.activations import *


tol = 1e-5
x = np.arange(-2, 2).reshape(2, 2)
sigmoid_x  = np.array([[0.11920292,  0.26894142],
                       [0.5,         0.73105858]])
dsigmoid_x = np.array([[0.10499359,  0.19661193],
                       [0.25,        0.19661193]])
relu_x     = np.array([[0.0,         0.0],
                       [0.0,         1.0]])
drelu_x    = np.array([[0.0,         0.0],
                       [0.0,         1.0]])
softmax_x  = np.array([[0.26894142, 0.73105858],
                       [0.26894142, 0.73105858]])
# dsoftmax_x = np.array([[0.19661193,  0.08914929],
#                        [-1.96183299, 0.19661193]])


def test_sigmoid():
    assert_allclose(Sigmoid.f(x),  sigmoid_x,  tol)
    assert_allclose(Sigmoid.df(x), dsigmoid_x, tol)


def test_relu():
    assert_allclose(ReLU.f(x),  relu_x,  tol)
    assert_allclose(ReLU.df(x), drelu_x, tol)


def test_softmax():
    assert_allclose(Softmax.f(x),  softmax_x,  tol)
    # assert_allclose(Softmax.df(x), dsoftmax_x, tol)
