import numpy as np
from nnlib.utils.data_utils import pad_with_zeros


def test_pad_with_zeros():
    array = np.arange(3*3*4*4).reshape(3, 3, 4, 4)
    padding = {2: 2, 3: 2}
    expected = np.zeros(shape=(3, 3, 6, 6), dtype=array.dtype)
    expected[:, :, 1:5, 1:5] = array
    got = pad_with_zeros(array, padding)
    assert (expected == got).all()
