"""Callbacks to be executed after each epoch of training."""
import os


class Callback:
    """Callback abstract base class - all callbacks should inherit from it."""
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class History(Callback):
    """Records metric values after each epoch of training.

    Metric values are kept in a dictionary of lists. If the History
    callback is provided fit method returns it after finishing training.

    Examples:
        To get a loss value of tenth epoch you'd need to do the following:
        >>>history = model.train(x_train, y_train, epochs=10, callbacks=[History()])
        >>>history["accuracy"][9]
    """
    def __init__(self):
        self.metrics = {}
        self._model = None

    def __getitem__(self, item):
        return self.metrics[item]

    def __call__(self, metrics, *_):
        for metric_identifier, metric in self.model.metrics.items():
            self.metrics[metric_identifier].append(metric.aggregate_value)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        for metric in self.model.metrics:
            self.metrics[metric] = []


class ModelCheckpoint(Callback):
    """Save the model after each epoch.

    Attributes:
        path: string, path to where to save the file
        monitor: string, performance metric to monitor
        save_best_only: bool, determines whether to overwrite the model
        verbose: bool, determines whether to notify the user that the
        model has been saved by printing to standard output
    """
    def __init__(self, path, monitor=None,
                 save_best_only=False, verbose=False):
        if not isinstance(monitor, str) and monitor is not None:
            raise TypeError("`monitor` should be a string identifier "
                            "of a metric tracked during training", monitor)
        if monitor is None and save_best_only:
            raise ValueError("Metric to be monitored is not specified!", self.monitor)

        self.path = path
        self.verbose = verbose
        self.save_best_only = save_best_only
        self._model = None
        self.monitor = monitor
        self.previous_metric_value = None

    def __call__(self, metrics, *_):
        metric = metrics[self.monitor]
        if self.save_best_only:
            if self.is_best(metric):
                self.save_model()
        else:
            self.save_model()
        self.previous_metric_value = metric.aggregate_value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        if self._model.metrics.get(self.monitor, None) is None:
            raise ValueError("Monitored metric is not measured "
                             "during training", self.monitor)

    def save_model(self):
        if self.verbose:
            if os.path.isabs(self.path):
                print("Saving model at location: {}.".format(self.path))
            else:
                print("Saving model at location: {}.".format(os.path.abspath(self.path)))
        self.model.save(self.path)

    def is_best(self, metric):
        if self.previous_metric_value is None:
            return True

        if metric.mode == "min":
            return self.previous_metric_value > metric.aggregate_value

        return self.previous_metric_value < metric.aggregate_value


class EarlyStopping(Callback):
    """Stops the training after a specified metric stops improving for a
    given number of epochs..

    Monitors specified metric. If the value of the specified metric
    continues to be equal or worse than a value previously recorded
    as the best one for the number of epochs specified by the patience
    attribute the model training will stop.

    Attributes:
        monitor: string, performance metric to monitor
        patience: int, number of epochs left for model to improve
        verbose: bool, determines whether to notify the user that the
        training terminated due to early stopping if that happens
    """

    def __init__(self, monitor=None, patience=0, verbose=False):
        if not isinstance(monitor, str):
            raise TypeError("`monitor` should be a string identifier "
                            "of a metric tracked during training", monitor)
        self.verbose = verbose
        self.monitor = monitor
        self.patience = patience
        self._model = None
        self.epochs_without_improvement = 0
        self.best_value = None

    def __call__(self, metrics, *_):
        metric = metrics[self.monitor]
        if self.improved(metric):
            self.best_value = metric.aggregate_value
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement > self.patience:
                if self.verbose:
                    print("Terminating training due to early stopping...")
                raise TerminateTraining

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        if self._model.metrics.get(self.monitor, None) is None:
            raise ValueError("Monitored metric is not measured "
                             "during training", self.monitor)

    def improved(self, metric):
        if self.best_value is None:
            return True

        if metric.mode == "min":
            return self.best_value > metric.aggregate_value

        return self.best_value < metric.aggregate_value


class LearningRateScheduler(Callback):
    """Schedules learning rate according to a defined procedure.

    Attributes:
        schedule: function, takes current learning rate and epoch number
        and returns new learning rate based on these two parameters
        verbose: bool, determines whether to notify the user that the
        learning rate is being changed by the scheduler
    """
    def __init__(self, schedule=None, verbose=False):
        self.schedule = schedule
        self.verbose = verbose
        self.model = None

    def __call__(self, metrics, epoch):
        lr = self.model.optimizer.lr

        try:
            lr = self.schedule(lr, epoch)
        except Exception:
            raise RuntimeError("Scheduling function didn't work as expected!")

        if not (isinstance(lr, int) or isinstance(lr, float)) or isinstance(lr, bool):
            raise TypeError("Scheduling function should return a float!", lr)

        self.model.optimizer.lr = lr
        if self.verbose:
            print("Scheduling learning rate to {}".format(lr))


class ReduceLROnPlateau(Callback):
    """Reduces learning rate after a specified metric stops improving
    for a specified number of epochs.

    Attributes:
        factor: float, factor by which to reduce the learning rate
        monitor: string, performance metric to monitor
        patience: int, number of epochs left for model to improve
        verbose: bool, determines whether to notify the user that the
        learning rate has been reduced by printing to the standard output
    """
    def __init__(self, monitor=None, factor=1.0, patience=0, verbose=False):
        if not isinstance(monitor, str):
            raise TypeError("`monitor` should be a string identifier "
                            "of a metric tracked during training", monitor)

        self.monitor = monitor
        self.patience = patience
        self.factor = factor
        self.verbose = verbose
        self._model = None
        self.epochs_without_improvement = 0
        self.best_value = None

    def __call__(self, metrics):
        metric = metrics[self.monitor]
        if self.improved(metric):
            self.best_value = metric.aggregate_value
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement > self.patience:
                self.model.optimizer.lr /= self.factor
                if self.verbose:
                    print("Reducing learning rate to {}"
                          .format(self.model.optimizer.lr))

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        if self._model.metrics.get(self.monitor, None) is None:
            raise ValueError("Monitored metric is not measured "
                             "during training", self.monitor)

    def improved(self, metric):
        if self.best_value is None:
            return True

        if metric.mode == "min":
            return self.best_value > metric.aggregate_value

        return self.best_value < metric.aggregate_value


class TerminateTraining(Exception):
    """Signals the train function to stop execution."""
    pass
