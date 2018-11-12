from __future__ import division
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
import torch


class MeanAbsoluteRelativeError(Metric):
    r"""
    Calculate Mean Absolute Relative Error

    :math:`\text{MARE} = \frac{1}{n}\sum _j^n\frac{\left|A_j-P_j\right|}{\left|A_j\right|}`
    Where, :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    More details can be found `here`_ .

    - `update` must receive output of the form `(y_pred, y)`

    .. _`here`:
        https://arxiv.org/ftp/arxiv/papers/1809/1809.03006.pdf
    """

    def reset(self):
        self._sum_of_absolute_relative_errors = 0.0
        self._num_samples = 0

    def update(self, output):
        y_pred, y = output
        if (y == 0).any():
            raise NotComputableError('The ground truth has 0')
        absolute_error = torch.abs(y_pred - y.view_as(y_pred)) / torch.abs(y.view_as(y_pred))
        self._sum_of_absolute_relative_errors += torch.sum(absolute_error).item()
        self._num_samples += y.shape[0]

    def compute(self):
        if self._num_samples == 0:
            raise NotComputableError('MeanAbsoluteRelativeError must have at least'
                                     'one sample before it can be computed')
        return self._sum_of_absolute_relative_errors / self._num_samples
