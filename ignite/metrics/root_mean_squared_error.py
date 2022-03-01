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

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in the format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``. If not, ``output_tranform`` can be added
        to the metric to transform the output into the form expected by the metric.

        ``y_pred`` and ``y`` should have the same shape.

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = RootMeanSquaredError()
            metric.attach(default_evaluator, 'rmse')
            preds = torch.Tensor([
                [1, 2, 4, 1],
                [2, 3, 1, 5],
                [1, 3, 5, 1],
                [1, 5, 1 ,11]
            ])
            target = preds * 0.75
            state = default_evaluator.run([[preds, target]])
            print(state.metrics['rmse'])

        .. testoutput::

            1.956559480312316
    """

    def compute(self) -> Union[torch.Tensor, float]:
        mse = super(RootMeanSquaredError, self).compute()
        return math.sqrt(mse)
