import numpy as np


class Regularizer:
    def __call__(self, x):
        return 0.


class L1(Regularizer):
    def __init__(self, lambda_=0.):
        self.lambda_ = lambda_

    def __call__(self, x):
        regularization = 0.
        regularization += np.sum(self.lambda_ * np.abs(x))
        return regularization


class L2(Regularizer):
    def __init__(self, lambda_=0.):
        self.lambda_ = lambda_

    def __call__(self, x):
        regularization = 0.
        regularization += np.sum(self.lambda_ * np.square(x))
        return regularization


class ElasticNet(Regularizer):
    def __init__(self, lambda_=0.):
        self.lambda_ = lambda_

    def __call__(self, x):
        regularization = 0.
        regularization += np.sum(self.lambda_ * np.abs(x) + (1 - self.lambda_) * np.square(x))
        return regularization


# Aliases
l1 = L1
l2 = L2
elastic_net = ElasticNet
