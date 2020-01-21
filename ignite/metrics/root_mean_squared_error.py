from __future__ import division
import math

from ignite.metrics.mean_squared_error import MeanSquaredError


class RootMeanSquaredError(MeanSquaredError):
    """
    Calculates the root mean squared error.

    - `update` must receive output of the form (y_pred, y) or `{'y_pred': y_pred, 'y': y}`.
    """
    def compute(self):
        mse = super(RootMeanSquaredError, self).compute()
        return math.sqrt(mse)
