from __future__ import division
import torch
from ignite.metrics.metric import Metric


class CanberraMetric(Metric):
    r"""
    Calculates the Canberra Metric.
    It has been proposed in `Botchkarev 2018`__.
    :math:`\text{CM} = \sum _j^n\frac{|A_j - P_j|}{A_j + P_j}`
    Where, :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.
    - `update` must receive output of the form `(y_pred, y)`.
    - `y` and `y_pred` must be of same shape.
    __ https://arxiv.org/abs/1809.03006
    """

    def reset(self):
        self._sum_of_errors = 0.0

    def update(self, output):
        y_pred, y = output
        errors = torch.abs(y.view_as(y_pred) - y_pred) / (y_pred + y.view_as(y_pred))
        self._sum_of_errors += torch.sum(errors).item()

    def compute(self):
        return self._sum_of_errors
