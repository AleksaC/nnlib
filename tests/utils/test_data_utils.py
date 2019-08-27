import numpy as np
from numpy.testing import assert_allclose

from nnlib.utils.data_utils import (
    one_hot,
    pad_with_zeros,
    shuffle_arrays,
    train_test_split
)


def test_one_hot():
    pass


def test_shuffle_arrays():
    x = np.arange(15)
    y = np.arange(15)
    z = np.arange(15)

    shuffle_arrays(x, y)

    assert (x == y).all()
    assert not (x == z).all()

    np.random.seed(0)
    np.random.shuffle(x)

    shuffle_arrays(y, seed=0)

    assert (x == y).all()


def test_pad_with_zeros():
    array = np.arange(3*3*4*4).reshape(3, 3, 4, 4)
    padding = {2: 2, 3: 2}

    expected = np.zeros(shape=(3, 3, 6, 6), dtype=array.dtype)
    expected[:, :, 1:5, 1:5] = array

    got = pad_with_zeros(array, padding)

    assert (expected == got).all()


def test_train_test_split():
    instances = np.random.uniform(size=(30, 5, 5, 3))
    labels = np.array([1, 2, 3] * 10)

    expected_train = (instances[:24], labels[:24])
    expected_test = (instances[24:], labels[24:])

    got_train, got_test = train_test_split(instances, labels, train_size=0.8, shuffle=False)

    assert_allclose(expected_train[0], got_train[0], rtol=1e-7)
    assert (expected_train[1] == got_train[1]).all()
    assert_allclose(expected_train[1], got_train[1], rtol=1e-7)
    assert (expected_test[1] == got_test[1]).all()

    np.random.seed(0)
    np.random.shuffle(instances)

    np.random.seed(0)
    np.random.shuffle(labels)

    expected_train = (instances[:24], labels[:24])
    expected_test = (instances[24:], labels[24:])

    got_train, got_test = train_test_split(instances, labels, train_size=0.8, seed=0)

    assert_allclose(expected_train[0], got_train[0], rtol=1e-7)
    assert (expected_train[1] == got_train[1]).all()
    assert_allclose(expected_train[1], got_train[1], rtol=1e-7)
    assert (expected_test[1] == got_test[1]).all()

    got_train, got_test = train_test_split(instances, labels, train_size=0.8, stratified=True)

    _, count = np.unique(got_train[1], return_counts=True)
    assert all(cnt == 8 for cnt in count)
