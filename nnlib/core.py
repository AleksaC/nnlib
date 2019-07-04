"""nnlib core module"""
from copy import deepcopy
import datetime
import pickle
import time
from warnings import warn

import numpy as np

from . import losses
from . import optimizers
from .callbacks import Callback
from .callbacks import History
from .callbacks import TerminateTraining
from .layers import Layer
from .metrics import Metric
from .metrics import binary_classification_accuracy
from .metrics import multilabel_classification_accuracy
from .utils.data_utils import BatchGenerator
from .utils.data_utils import shuffle_arrays
from .utils.data_utils import to_numpy_array
from .utils.logging import ProgressBar
from .utils.logging import log_metrics


class Model:
    """Core model class

    Attributes:
        layers: list, contains instances of layers.Layer
        loss: losses.Loss
        optimizer: optimizers.Optimizer
        metrics: dictionary, contains performance metrics to be monitored
        during training and testing

    """
    def __init__(self, *layers, loss=None, optimizer=None, metrics=None):
        self._loss = None
        self._optimizer = None
        self.layers = []
        self.trainable_layers = {}
        self.metrics = {}

        if layers is not None:
            input_shape = None
            for layer in layers:
                input_shape = self._add(layer, input_shape)

        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer
            self._optimizer.model = self

        if self.layers is not None:
            output_shape = self.layers[-1].output_shape()
            if output_shape[0] == 1:
                self.metrics["accuracy"] = binary_classification_accuracy
            else:
                self.metrics["accuracy"] = multilabel_classification_accuracy

        if metrics is not None:
            if not (isinstance(metrics, list) or isinstance(metrics, tuple)):
                raise TypeError("", metrics)
            for metric in metrics:
                if isinstance(metric, str):
                    self.metrics[metric] = Metric(metric)
                elif isinstance(metric, Metric):
                    self.metrics[metric.__class__.__name__] = metric
                else:
                    raise TypeError("...", metric)

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, loss):
        if isinstance(loss, str):
            self._loss = losses.get_loss(loss)
        elif isinstance(loss, losses.Loss):
            self._loss = loss
        else:
            raise TypeError("Loss function should be specified either "
                            "by passing a subclass of abstract class `Loss` "
                            "or by passing a string identifier of the loss!", loss)

        self.metrics["loss"] = Metric(self._loss.f, mode="min")

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if isinstance(optimizer, str):
            self._optimizer = optimizers.get_optimizer(optimizer)
        elif isinstance(optimizer, optimizers.Optimizer):
            self._optimizer = optimizer
        else:
            raise TypeError("Optimizer should be specified either by "
                            "passing a subclass of abstract class `Optimizer` "
                            "or by passing a string identifier of the optimizer!", optimizer)

        self._optimizer.model = self

    def _assert_initialized(self):
        if self.loss is None:
            raise ValueError("Loss function needs to be specified before using the model!")
        if self.optimizer is None:
            raise ValueError("Optimizer needs to be specified before using the model")
        if self.layers is None:
            raise ValueError("Layers need to be initialized before using the model")

    def _add(self, layer, input_shape=None):
        if isinstance(layer, Layer):
            if input_shape is None:
                if layer.input_shape is None:
                    raise ValueError("First layer should have attribute "
                                     "`input_shape` specified", layer)
                else:
                    input_shape = layer.input_shape
            layer.initialize(input_shape)
            self.layers.append(layer)

            if layer.trainable:
                self.trainable_layers[layer.uuid] = layer
        else:
            raise TypeError("Layers should be specified by passing an "
                            "instance of a `layers.Layer` class", layer)

        return layer.output_shape()

    def save(self, path=None):
        if path is None:
            path = "model"

        with open(path, "wb") as f:
            pickle.dump(self, f)

    def summary(self):
        """Prints model summary.

        The summary includes layer names, their input and output shape,
        as well as number of parameters.
        """
        print("\nModel summary:")
        print("===============================================================================================")
        number_of_params = 0
        for layer in self.layers:
            number_of_params += layer.number_of_parameters()
            print(layer)
            print("===============================================================================================")
        print("Total parameters: {:,}".format(number_of_params))
        print()

    def _forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def _forward_inference(self, x):
        for layer in self.layers:
            x = layer.forward_inference(x)
        return x

    def _backward(self, grad):
        grads = {}
        for layer in reversed(self.layers):
            grad, grads[layer.uuid] = layer.backward(grad)
        grads = dict((layer, grad) for layer, grad in grads.items() if grad is not None)
        return grads

    def _train_on_batch(self, x, y):
        grads, y_pred = self.optimizer.compute_grads(x, y)
        self.optimizer.update(self.trainable_layers, grads)
        return y_pred

    def _train_on_epoch(self, x, y, batch_size, logging_level):
        iteration = 0

        generator = BatchGenerator(x, y, batch_size)
        if logging_level == 2:
            generator = ProgressBar(generator)
            generator.metrics = self.metrics

        for x_batch, y_batch in generator:
            y_pred = self._train_on_batch(x_batch, y_batch)
            iteration += 1
            for metric in self.metrics.values():
                metric(y_batch, y_pred)

        for metric in self.metrics.values():
            if not metric.is_global:
                metric.aggregate_value /= iteration

        if logging_level > 0:
            print()

    def train(self, x, y, batch_size=32, epochs=1, logging_level=0,
              shuffle=True, callbacks=None, validation_data=None):
        """Method for training the model.

        Args:
            x: training data
            y: training labels
            batch_size:
            epochs: number of iterations over entire dataset
            logging_level: int, level of logging one of {0, 1, 2}, 0 - no logging,
            1 - logging after each epoch, 2 - logging periodically with progress bar
            shuffle: bool, specifies whether to shuffle the input arrays after each epoch
            callbacks: list, contains instances of Callback methods
            validation_data: tuple, (x_val, y_val)
        """
        self._assert_initialized()

        if x.shape[0] != y.shape[0]:
            raise ValueError("The number of data points and corresponding labels "
                             "are not the same", x.shape[0], y.shape[0])

        x = to_numpy_array(x)
        y = to_numpy_array(y)

        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise ValueError("`batch_size` should be an integer specifying the number of iterations "
                             "over the entire dataset, got {}.".format(batch_size))

        if not isinstance(epochs, int) or isinstance(epochs, bool):
            raise ValueError("`epochs` should be an integer specifying the number of iterations "
                             "over the entire dataset, got {}.".format(epochs))

        if not isinstance(logging_level, int) or isinstance(logging_level, bool) or logging_level > 2:
            print()
            warn("`logging_level` should be one of {{1, 2, 3}}, got {}. Logging level "
                 "will default to 0, i.e. no logging will take place".format(logging_level))
            logging_level = 0

        if not isinstance(shuffle, bool):
            warn("`shuffle` should be a bool, got {]. Will default to True.".format(logging_level))

        if validation_data is not None:
            if not isinstance(validation_data, tuple):
                raise TypeError("validation_data should be a tuple of the "
                                "form (x_val, y_val).", validation_data)
            if len(validation_data) != 2:
                raise ValueError("validation_data should be a tuple of the "
                                 "form (x_val, y_val).", validation_data)
            if validation_data[0].shape[0] != validation_data[1].shape[0]:
                raise ValueError("The number of validation data points and corresponding labels "
                                 "are not the same", x.shape[0], y.shape[0])

        batch_size = min(batch_size, len(x))

        history = None

        if callbacks is None:
            callbacks = []
        elif not (isinstance(callbacks, tuple) or isinstance(callbacks, list)):
            raise TypeError("Callbacks should be provided as a list or tuple "
                            "of instances of `callbacks.Callback`", callbacks)
        else:
            for callback in callbacks:
                if not isinstance(callback, Callback):
                    raise TypeError("Callbacks should be provided as a list or tuple "
                                    "of instances of `callbacks.Callback`", callback)
                callback.model = self
                if isinstance(callback, History):
                    history = callback

        time_started = time.time()

        for epoch in range(epochs):
            if logging_level > 0:
                epoch_counter = "Epoch {}:".format(epoch + 1)
                print(epoch_counter)
                print("="*len(epoch_counter))

            if shuffle:
                shuffle_arrays(x, y)

            self._train_on_epoch(x, y, batch_size, logging_level)

            metrics = self.metrics
            if validation_data is not None:
                validation_score = self.test(*validation_data)
                validation_score = dict(("val_" + score, validation_score[score]) for score in validation_score)

                if logging_level > 0:
                    print("\nValidation results: ")
                    log_metrics(validation_score, inline=False)

                metrics = dict(metrics, **validation_score)

            for callback in callbacks:
                try:
                    callback(metrics, epoch)
                except TerminateTraining:
                    return history

            for metric in self.metrics.values():
                metric.aggregate_value = 0

        time_elapsed = time.time() - time_started
        time_elapsed = datetime.timedelta(seconds=round(time_elapsed))
        print("Finished training. Total training time: {}\n".format(time_elapsed))

        return history

    def predict(self, x, batch_size=32, logging_level=0):
        """Makes predictions for the provided inputs

        Args:
            x: tensor, input data
            batch_size: int, size of batches to split data into
            logging_level: int, level of logging, valid values {0, 1}

        Returns:
            tensor, contains predictions for the provided input
        """
        x = np.array(x, dtype="float32")

        batch_size = min(batch_size, len(x))

        if not isinstance(logging_level, int) or isinstance(logging_level, bool):
            warn("`logging_level` should be either 0 or 1, got {}. Logging level "
                 "will default to 0, i.e. no logging will take place".format(logging_level))
        elif logging_level != 0 and logging_level != 1:
            warn("`logging_level` should be either 0 or 1, got: {}. Logging level "
                 "will default to 0, i.e. no logging will take place".format(logging_level))

        batch = 0
        y_pred = np.empty((x.shape[0],) + self.layers[-1].output_shape())

        if logging_level == 1:
            progbar = ProgressBar()
            print("Running inference:")

        while batch < len(x):
            if logging_level == 1:
                progbar(batch, len(x))
            y_pred[batch : batch+batch_size] = self._forward_inference(x[batch : batch+batch_size])
            batch += batch_size

        return y_pred

    def test(self, x, y, batch_size=32, logging_level=0):
        """Evaluates model performance for a given input.

        Args:
            x: tensor, input data
            y: tensor, labels
            batch_size: int, size of batches to split data into
            logging_level: int, level of logging, valid values {0, 1}

        Returns:
            tuple of values of specified metrics
        """
        self._assert_initialized()

        x = to_numpy_array(x)
        y = to_numpy_array(y)

        if x.shape[0] != y.shape[0]:
            raise ValueError("The number of data points and corresponding labels "
                             "are not the same", x.shape[0], y.shape[0])

        batch_size = min(batch_size, len(x))

        y_pred = self.predict(x, batch_size=batch_size)

        metrics = deepcopy(self.metrics)
        for metric in metrics.values():
            metric(y, y_pred)

        if not isinstance(logging_level, int) or isinstance(logging_level, bool):
            warn("`logging_level` should be either 0 or 1, got {}. Logging level "
                 "will default to 0, i.e. no logging will take place".format(logging_level))
        elif logging_level == 1:
            print("Evaluation results:")
            log_metrics(metrics, inline=False)
        elif logging_level != 0:
            warn("`logging_level` should be either 0 or 1, got: {}. Logging level "
                 "will default to 0, i.e. no logging will take place".format(logging_level))

        return metrics


def load_model(path):
    """Loads model stored in a specified location.

    Args:
        path: str, relative or absolute path to the saved model

    Returns:
        nnlib Model instance

    Raises:
        ValueError if the file specified does not exist
        or cannot be unpickled
    """
    try:
        f = open(path, "rb")
        model = pickle.load(f)
    except FileNotFoundError:
        raise ValueError("Model is not found at specified location, "
                         "file specified does not exist.", path)
    except pickle.UnpicklingError:
        raise ValueError("Couldn't unpickle the file, it is either "
                         "corrupted or not a valid pickle file")
    finally:
        f.close()

    if not isinstance(model, Model):
        raise ValueError("The specified file doesn't contain a nnlib model")

    return model
