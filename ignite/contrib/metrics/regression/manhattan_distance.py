from typing import Tuple

import torch

from ignite.contrib.metrics.regression._base import _BaseRegression
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class ManhattanDistance(_BaseRegression):
    r"""Calculates the Manhattan Distance.

    .. math::
        \text{MD} = \sum_{j=1}^n |A_j - P_j|

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    More details can be found in `scikit-learn distance metrics`__.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)`.

    __ https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html

    Parameters are inherited from ``Metric.__init__``.

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
        The output of the engine's ``process_function`` needs to be in format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``.

        .. testcode::

            metric = ManhattanDistance()
            metric.attach(default_evaluator, 'manhattan')
            y_true = torch.Tensor([0, 1, 2, 3, 4, 5])
            y_pred = y_true * 0.75
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['manhattan'])

        .. testoutput::

            3.75...

    .. versionchanged:: 0.4.3

        - Fixed sklearn compatibility.
        - Workes with DDP.
    """

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_errors = torch.tensor(0.0, device=self._device)

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output
        errors = torch.abs(y - y_pred)
        self._sum_of_errors += torch.sum(errors).to(self._device)

    @sync_all_reduce("_sum_of_errors")
    def compute(self) -> float:
        return self._sum_of_errors.item()
