from __future__ import division
import math

import torch

from .mean_squared_error import MeanSquaredError
from ignite.exceptions import NotComputableError


class RootMeanSquaredError(MeanSquaredError):
    """
    Calculates the root mean squared error.

    `update` must receive output of the form (y_pred, y).
    """
    def compute(self):
        mse = super(RootMeanSquaredError, self).compute()
        return math.sqrt(mse)
