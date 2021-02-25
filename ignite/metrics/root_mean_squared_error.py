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

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
    """

    def compute(self) -> Union[torch.Tensor, float]:
        mse = super(RootMeanSquaredError, self).compute()
        return math.sqrt(mse)
