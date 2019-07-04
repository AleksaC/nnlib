"""Various utility functions"""
import numpy as np


def get_top_k(y_pred, k):
    """Finds indices of top k output probabilities of a classifier.

    Args:
        y_pred: tensor, classifier predictions
        k: int, number of top probabilities to consider

    Returns:
        tensor, contains indices of top k predictions of a classifier
    """
    top_k_indices = np.argpartition(y_pred, -k, axis=-1)[:, -k:]
    return top_k_indices


def get_from_module(module_params, module_name, identifier):
    """Gets a class/instance of a module member specified by the identifier.

    Args:
        module_params: dict, contains identifiers
        module_name: str, containing the name of the module
        identifier: str, specifying the module member

    Returns:
        a class or an instance of a module member specified
        by the identifier
    """
    res = module_params.get(identifier.lower())
    if res is None:
        raise ValueError("Invalid {} identifier!".format(module_name), identifier)
    return res
