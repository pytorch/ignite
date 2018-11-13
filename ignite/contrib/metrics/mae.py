from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
import torch


class MaximumAbsoluteError(Metric):
    r"""
    Calculates the Maximum Absolute Error.

    It has been proposed in `Performance Metrics (Error Measures) in Machine Learning Regression, Forecasting and
    Prognostics: Properties and Typology`.

    More details can be found in `here`_.

    :math:`\text{MaxAE} = \max_{j=1,n} \left(|A_j-P_j\right|)`

    Where, :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y` and `y_pred` must be of same shape.

    .. _here:
        https://arxiv.org/ftp/arxiv/papers/1809/1809.03006.
    """

    def reset(self):
        self._max_of_absolute_errors = -1

    def update(self, output):
        y_pred, y = output
        mae = torch.abs(y_pred - y.view_as(y_pred)).max().item()
        if self._max_of_absolute_errors < mae:
            self._max_of_absolute_errors = mae

    def compute(self):
        if self._max_of_absolute_errors < 0:
            raise NotComputableError('MaximumAbsoluteError must have at least one example before it can be computed')
        return self._max_of_absolute_errors
