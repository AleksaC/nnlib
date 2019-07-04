"""Performance metrics"""

import numpy as np

from .utils.generic_utils import get_top_k


class Metric:
    def __init__(self, f, mode="auto"):
        self.f = f
        self.current_value = None
        self.aggregate_value = 0
        self.is_global = False
        self.mode = mode

    def __call__(self, y, y_pred):
        self.current_value = self.f(y, y_pred)
        self.aggregate_value += self.current_value
        return self.current_value


class Global:
    def __init__(self, metric):
        metric.is_global = True

    def __call__(self, *args, **kwargs):
        pass


def _multilabel_classification_accuracy(y, y_pred):
    """Computes accuracy of predictions in multilabel classification.

    Args:
        y: tensor, true labels
        y_pred: tensor, predicted labels

    Returns:
        float, represents percentage of correct predictions
    """
    return np.sum(np.argmax(y_pred, axis=-1) == np.argmax(y, axis=-1)) / y.shape[0]


def _binary_classification_accuracy(y, y_pred):
    """Computes accuracy of predictions of a binary classifier.

    Args:
        y: tensor, true labels
        y_pred: tensor, predicted labels

    Returns:
        float, represents percentage of correct predictions
    """
    return np.sum(np.where(y_pred > 0.5, 1, 0) == y) / len(y)


def _top_k_accuracy(y, y_pred, k=5):
    """Calculates the top-k accuracy of the model.

    Args:
        y: tensor, true labels
        y_pred: tensor, model predictions
        k: int, how many top predictions to consider

    Returns:
        float, represents percentage of times the correct label was
        in top k predictions
    """
    top_k = get_top_k(y_pred, k)
    correct = 0

    for i, y_i in enumerate(np.argmax(y, axis=-1)):
        if y_i in top_k[i]:
            correct += 1

    return correct / y.shape[0]


@Global
def precision(y, y_pred):
    raise NotImplementedError


@Global
def recall(y, y_pred):
    raise NotImplementedError


@Global
def f_score(y, y_pred, beta=1):
    prec = precision(y, y_pred)
    rec = recall(y, y_pred)

    return (1 + beta**2) * prec * rec / (beta**2 * prec + rec)


multilabel_classification_accuracy = Metric(_multilabel_classification_accuracy, mode="max")
binary_classification_accuracy = Metric(_binary_classification_accuracy, mode="max")
top_k_accuracy = Metric(_top_k_accuracy, mode="max")
