from __future__ import division

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


class ManhattanDistance(Metric):
    r"""
    Calculates the Manhattan Distance.

    It has been proposed in `Performance Metrics (Error Measures) in Machine Learning Regression, Forecasting and
    Prognostics: Properties and Typology`.

    More details can be found in `here`_.

    :math:`\text{MD} = \sum _j^n (A_j - P_j)`

    Where, :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y` and `y_pred` must be of same shape.

    .. _here:
        https://arxiv.org/ftp/arxiv/papers/1809/1809.03006.pdf
    """
    def reset(self):
        self._sum_of_errors = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        errors = y_pred - y.view_as(y_pred)
        self._sum_of_errors += torch.sum(errors).item()
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('ManhattanDistance must have at least one example before it can be computed')
        return self._sum_of_errors
