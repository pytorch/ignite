from typing import Tuple

import torch

from ignite.contrib.metrics.regression._base import _BaseRegression
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class MeanNormalizedBias(_BaseRegression):
    r"""Calculates the Mean Normalized Bias.

    .. math::
        \text{MNB} = \frac{1}{n}\sum_{j=1}^n\frac{A_j - P_j}{A_j}

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    More details can be found in the reference `Botchkarev 2018`__.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)`.

    __ https://arxiv.org/abs/1809.03006

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

            metric = MeanNormalizedBias()
            metric.attach(default_evaluator, 'mnb')
            y_true = torch.Tensor([1, 2, 3, 4, 5])
            y_pred = y_true * 0.75
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['mnb'])

        .. testoutput::

            0.25...

    .. versionchanged:: 0.4.5
        - Works with DDP.
    """

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_errors = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()

        if (y == 0).any():
            raise NotComputableError("The ground truth has 0.")

        errors = (y.view_as(y_pred) - y_pred) / y
        self._sum_of_errors += torch.sum(errors).to(self._device)
        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_of_errors", "_num_examples")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("MeanNormalizedBias must have at least one example before it can be computed.")
        return self._sum_of_errors.item() / self._num_examples
