import math
from typing import Union

import torch

from ignite.metrics.mean_squared_error import MeanSquaredError

__all__ = ["RootMeanSquaredError"]


class RootMeanSquaredError(MeanSquaredError):
    """
    Calculates the root mean squared error.

    - ``update`` must receive output of the form (y_pred, y) or `{'y_pred': y_pred, 'y': y}`.
    """

    def compute(self) -> Union[torch.Tensor, float]:
        mse = super(RootMeanSquaredError, self).compute()
        return math.sqrt(mse)
