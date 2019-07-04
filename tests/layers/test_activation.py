import pytest
import numpy as np

from nnlib.layers import *


def test_prelu():
    mock_input = np.ones((5, 5))
    mock_input[0, 0] = mock_input[1, 1] = -1

    mock_output = np.ones_like(mock_input)
    mock_output[0, 0] = mock_output[1, 1] = -0.25

    prelu = PReLU()
    prelu.initialize(mock_input.shape)
    prelu.x = mock_input

    assert (prelu.f(mock_input) == mock_output).all()

    mock_output = np.zeros_like(mock_input)
    mock_output[0, 0] = mock_output[1, 1] = -1

    assert (prelu.df(mock_input) == mock_output).all()
