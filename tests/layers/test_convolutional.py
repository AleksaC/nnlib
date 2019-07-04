import numpy as np
from nnlib.layers.convolutional import im2col


def test_im2col_shape():
    imgs = np.zeros((1, 3, 227, 227))
    a = im2col(imgs, 11, 4)
    assert a.shape == (1, 363, 3025)


def test_im2col():
    imgs = np.stack((np.arange(48).reshape((3, 4, 4)),)*3)
    expected = np.array([
        [ 0,  2,  8, 10],
        [ 1,  3,  9, 11],
        [ 4,  6, 12, 14],
        [ 5,  7, 13, 15],
        [16, 18, 24, 26],
        [17, 19, 25, 27],
        [20, 22, 28, 30],
        [21, 23, 29, 31],
        [32, 34, 40, 42],
        [33, 35, 41, 43],
        [36, 38, 44, 46],
        [37, 39, 45, 47]
    ])
    expected = np.stack((expected,)*3)
    result = im2col(imgs, 2, 2)

    assert (expected == result).all()
