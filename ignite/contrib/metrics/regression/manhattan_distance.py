from __future__ import division

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


class ManhattanDistance(Metric):
    r"""
    Calculates the Manhattan Distance.

    It has been proposed in `Botchkarev 2018`__.

    More details can be found in `https://arxiv.org/ftp/arxiv/papers/1809/1809.03006.pdf`.

    :math:`\text{MD} = \sum _j^n (A_j - P_j)`

    Where, :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y` and `y_pred` must be of same shape.

    __ https://arxiv.org/abs/1809.03006

    """
    def reset(self):
        self._sum_of_errors = 0.0

    def update(self, output):
        y_pred, y = output
        errors = y_pred - y.view_as(y_pred)
        self._sum_of_errors += torch.sum(errors).item()

    def compute(self):
        return self._sum_of_errors
