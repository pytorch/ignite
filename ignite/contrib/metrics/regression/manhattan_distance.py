from __future__ import division

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


class ManhattanDistance(Metric):
    r"""
    Calculates the Manhattan Distance:

    :math:`\text{MD} = \sum_{j=1}^n (A_j - P_j)`,

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    More details can be found in `Botchkarev 2018`__.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y` and `y_pred` must be of same shape.

    __ https://arxiv.org/abs/1809.03006

    """
    def reset(self):
        self._sum_of_errors = 0.0

    def update(self, output):
        y_pred, y = output
        errors = y.view_as(y_pred) - y_pred
        self._sum_of_errors += torch.sum(errors).item()

    def compute(self):
        return self._sum_of_errors
