"""The MNIST dataset"""
import os
from warnings import warn

import numpy as np

from ..utils import data_utils
from ..config import base_dir


def _download_data(url):
    """Downloads the mnist data from specified url.

    Args:
        url:

    Returns:

    """
    data = data_utils.download_data(url)
    magic_number = int.from_bytes(data[:4], byteorder="big", signed=False)

    if "labels" in url:
        if magic_number != 2049:
            raise ValueError("Wrong magic number ({}) in file {}!"
                             .format(magic_number, url))
        data = np.frombuffer(data[8:], dtype="uint8")
    else:
        if magic_number != 2051:
            raise ValueError("Wrong magic number ({}) in file {}!"
                             .format(magic_number, url))
        data = np.frombuffer(data[16:], dtype="uint8").reshape(-1, 28, 28)

    return data


def _load_data(data, labels, scaled, one_hot_labels,
               flat, cache, cache_dir=None):
    """Loads MNIST data and labels either from cache or by downloading it.

    """
    base_url = "http://yann.lecun.com/exdb/mnist/"

    if not cache and cache_dir is not None:
        warn("Caching is disabled, but the cache directory is specified")

    if cache_dir is None:
        cache_dir = os.path.join(base_dir, "data", "mnist")

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    destination = "training.npz" if "train" in data else "testing.npz"
    destination = os.path.join(cache_dir, destination)

    if os.path.exists(destination):
        mnist = np.load(destination)
        data = mnist["data"]
        labels = mnist["labels"]
    else:
        data = _download_data(base_url + data)
        labels = _download_data(base_url + labels)

    if cache:
        np.savez(destination, data=data, labels=labels)

    if scaled:
        data = data / 255
    if one_hot_labels:
        labels = data_utils.one_hot(labels, 10)
    if flat:
        data = data.reshape(-1, 784)

    return data, labels


def training_data(scaled=True, one_hot_labels=True, flat=False,
                  cache=True, cache_dir=None):
    """Loads preprocessed MNIST training data.

    Args:
        scaled: bool, determines whether to normalize the data by dividing
        each pixel value by 255
        one_hot_labels: bool, determines whether to return one-hot encoded
        labels instead of the integer ones
        flat: bool, determines whether to flatten the images into
        one-dimensional arrays
        cache: bool, determines whether to save the data locally once it
        has been downloaded
        cache_dir: str, path to the directory where the cached data
        should be stored

    Returns:
        tuple of numpy ndarrays: (x_train, y_train) where x_train represents
        the training data and y_train represents the corresponding labels
    """
    return _load_data(data="train-images-idx3-ubyte.gz",
                      labels="train-labels-idx1-ubyte.gz",
                      scaled=scaled,
                      one_hot_labels=one_hot_labels,
                      flat=flat,
                      cache=cache,
                      cache_dir=cache_dir)


def test_data(scaled=True, one_hot_labels=True, flat=False,
              cache=True, cache_dir=None):
    """Loads preprocessed MNIST test data.

    Args:
        scaled: bool, determines whether to normalize the data by dividing
        each pixel value by 255
        one_hot_labels: bool, determines whether to return one-hot encoded
        labels instead of the integer ones
        flat: bool, determines whether to flatten the images into
        one-dimensional arrays
        cache: bool, determines whether to save the data locally once it
        has been downloaded
        cache_dir: str, path to the directory where the cached data
        should be stored

    Returns:
        tuple of numpy ndarrays: (x_test, y_test) where x_test represents
        the testing data and y_test represents the corresponding labels

    """
    return _load_data(data="t10k-images-idx3-ubyte.gz",
                      labels="t10k-labels-idx1-ubyte.gz",
                      scaled=scaled,
                      one_hot_labels=one_hot_labels,
                      flat=flat,
                      cache=cache,
                      cache_dir=cache_dir)
