"""Various utilities for loading, saving and pre-processing data"""
import math
import zlib
from urllib.request import urlopen
from warnings import warn

import numpy as np


def _is_pandas_dataframe(x):
    try:
        from pandas import DataFrame
    except ImportError:
        return False
    return type(x) == DataFrame


def to_numpy_array(data):
    """Creates a numpy array containing provided numerical data.

    If provided data is already a numpy array it will be returned
    without modifications. If it is a pandas dataframe data.values
    will be returned. Otherwise it will be attempted to create a
    numpy array from the data using np.array(). If this fails to
    provide an array whose elements are of type np.number a TypeError
    will be raised.

    Args:
        data: object, contains numeric data

    Returns:
        tensor, contains provided data converted into a numpy array
    Raises:
        TypeError when data cannot be converted to a numpy array
    """
    if type(data) == np.ndarray:
        return data

    if _is_pandas_dataframe(data):
        return data.values

    data = np.array(data, dtype="float32")
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("Only data containing numeric values"
                        "can be converted to numpy array using"
                        "this function", data)

    return data


def write_data(data, fpath, mode="wb"):
    with open(fpath, mode) as f:
        f.write(data)


def _decompress(data):
    return zlib.decompress(data, 32 + zlib.MAX_WBITS)


def download_data(url, decompress=True):
    """Downloads data from a given URL and places it into a specified file.

    """
    data = urlopen(url).read()
    if decompress:
        data = _decompress(data)
    return data


def save_file(origin, destination, decompress=True):
    data = download_data(origin, decompress=decompress)
    write_data(data, destination)
    return data


def read_data(fpath, decompress=False):
    with open(fpath, "rb") as f:
        data = f.read()

    if decompress:
        data = _decompress(data)

    return data


def pad_with_zeros(array, padding_by_axis):
    """Pads array with zeros by a given amount over each axis.

    Args:
        array: array to be padded
        padding_by_axis: dict, keys correspond to axes and values to the padding
        to be added for the given axis

    Returns:
        Padded array
    """
    shape = [
        size + padding_by_axis.get(axis, 0)
        for axis, size in enumerate(array.shape)
    ]

    position = tuple(
        slice(padding_by_axis.get(axis, 0) // 2,
              size - (padding_by_axis.get(axis, 0) + 1) // 2, 1)
        for axis, size in enumerate(shape)
    )

    padded_array = np.zeros(shape=shape, dtype=array.dtype)
    padded_array[position] = array

    return padded_array


def shuffle_arrays(*arrays, seed=None):
    """Shuffles in-place each array in a tuple of numpy arrays in
    the same order along the first axis.

    Args:
        *arrays: tuple, contains numpy arrays to be shuffled
        seed: int, seed for the random number generator
    """
    if not all(arr.shape[0] == arrays[0].shape[0] for arr in arrays):
        warn("Arrays provided to `shuffle_arrays` function are not of the same shape")

    if seed is not None:
        np.random.seed(seed)

    state = np.random.get_state()
    for array in arrays:
        np.random.set_state(state)
        np.random.shuffle(array)


def one_hot(y, num_classes):
    """Performs one-hot encoding of the labels.

    Args:
        y: tensor, labels
        num_classes: int, number of classes

    Returns:
        tensor, one-hot encoded labels, has shape
        (num_labels, num_classes)
    """
    return np.eye(num_classes)[y]


def train_test_split(instances, labels, train_size,
                     shuffle=True, stratified=False, seed=None):
    """Splits the dataset into training and testing sets.

    Args:
        instances:
        labels:
        train_size:
        shuffle:
        stratified:
        seed:

    Returns:
        tuple of tuples, first tuple contains training data while the second
        contains testing data - (x_train, y_train), (x_test, y_test)

    """
    if instances.shape[0] != labels.shape[0]:
        raise ValueError("Arrays containing instances and labels don't have the "
                         "same shape along the first axis!", instances, labels)

    if isinstance(train_size, int):
        if train_size > instances.shape[0]:
            raise ValueError("`train_size` cannot be greater than the size of "
                             "the entire dataset! ")
        training_samples = train_size
        train_size /= instances.shape[0]
    elif isinstance(train_size, float):
        if train_size < 0 or train_size > 1:
            raise ValueError("`train_size` should be a float between 0 and 1 "
                             "corresponding to percentage of the entire dataset "
                             "to be used as a training set, or an int specifying "
                             "exactly how many samples to be used in the training "
                             "set. Got: {}")
        training_samples = int(round(train_size * instances.shape[0]))
    else:
        raise TypeError("`train_size` should be a float between 0 and 1 "
                        "corresponding to percentage of the entire dataset "
                        "to be used as a training set, or an int specifying "
                        "exactly how many samples to be used in the training "
                        "set. Got: {}")

    if seed is not None:
        np.random.seed(seed)

    if shuffle:
        state = np.random.get_state()
        np.random.shuffle(instances)
        np.random.set_state(state)
        np.random.shuffle(labels)

    if not shuffle and stratified:
        warn("shuffle is set to False while stratify is set to True. Stratified "
             "split cannot be performed without shuffling the original arrays.")

    if stratified:
        classes, num_instances = np.unique(labels, return_counts=True)
        counter = {
            label: int(round(count * train_size))
            for label, count in zip(classes, num_instances)
        }
        test_samples = instances.shape[0] - training_samples

        train_instances = np.empty(shape=(training_samples,) + instances.shape[1:],
                                   dtype=instances.dtype)
        test_instances  = np.empty(shape=(test_samples,) + instances.shape[1:],
                                   dtype=instances.dtype)
        train_labels    = np.empty(shape=training_samples, dtype=labels.dtype)
        test_labels     = np.empty(shape=test_samples, dtype=labels.dtype)

        i, j = 0, 0

        for instance, label in zip(instances, labels):
            if counter[label] > 0:
                train_instances[i] = instance
                train_labels[i] = label
                counter[label] -= 1
                i += 1
            else:
                test_instances[j] = instance
                test_labels[j] = label
                j += 1
    else:
        train_instances = instances[:training_samples]
        test_instances  = instances[training_samples:]
        train_labels    = labels[:training_samples]
        test_labels     = labels[training_samples:]

    return (train_instances, train_labels), (test_instances, test_labels)


class BatchGenerator:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.stop = math.ceil(x.shape[0] / batch_size)
        self._generator = self.__call__()

    def __call__(self):
        batch = 0
        while batch < self.x.shape[0]:
            yield self.x[batch : batch+self.batch_size], \
                  self.y[batch : batch+self.batch_size]
            batch += self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._generator)
