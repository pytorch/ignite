from typing import Tuple

import torch

from ignite.contrib.metrics.regression._base import _BaseRegression
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class R2Score(_BaseRegression):
    r"""Calculates the R-Squared, the
    `coefficient of determination <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_.

    .. math::
        R^2 = 1 - \frac{\sum_{j=1}^n(A_j - P_j)^2}{\sum_{j=1}^n(A_j - \bar{A})^2}

    where :math:`A_j` is the ground truth, :math:`P_j` is the predicted value and
    :math:`\bar{A}` is the mean of the ground truth.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)` and of type `float32`.

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

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = R2Score()
            metric.attach(default_evaluator, 'r2')
            y_true = torch.tensor([0., 1., 2., 3., 4., 5.])
            y_pred = y_true * 0.75
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['r2'])

        .. testoutput::

            0.8035...

    .. versionchanged:: 0.4.3
        Works with DDP.
    """

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_examples = 0
        self._sum_of_errors = torch.tensor(0.0, device=self._device)
        self._y_sq_sum = torch.tensor(0.0, device=self._device)
        self._y_sum = torch.tensor(0.0, device=self._device)

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output
        self._num_examples += y.shape[0]
        self._sum_of_errors += torch.sum(torch.pow(y_pred - y, 2)).to(self._device)

        self._y_sum += torch.sum(y).to(self._device)
        self._y_sq_sum += torch.sum(torch.pow(y, 2)).to(self._device)

    @sync_all_reduce("_num_examples", "_sum_of_errors", "_y_sq_sum", "_y_sum")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("R2Score must have at least one example before it can be computed.")
        return 1 - self._sum_of_errors.item() / (self._y_sq_sum.item() - (self._y_sum.item() ** 2) / self._num_examples)
