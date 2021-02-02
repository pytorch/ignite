import math
from typing import Union

import torch

from ignite.metrics.mean_squared_error import MeanSquaredError

__all__ = ["RootMeanSquaredError"]


class RootMeanSquaredError(MeanSquaredError):
    r"""Calculates the `root mean squared error <https://en.wikipedia.org/wiki/Root-mean-square_deviation>`_.

    .. math:: \text{RMSE} = \sqrt{ \frac{1}{N} \sum_{i=1}^N \left(y_{i} - x_{i} \right)^2 }

    where :math:`y_{i}` is the prediction tensor and :math:`x_{i}` is ground true tensor.

    - ``update`` must receive output of the form (y_pred, y) or `{'y_pred': y_pred, 'y': y}`.
    """

    def compute(self) -> Union[torch.Tensor, float]:
        mse = super(RootMeanSquaredError, self).compute()
        return math.sqrt(mse)
