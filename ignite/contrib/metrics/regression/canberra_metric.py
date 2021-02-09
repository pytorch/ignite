from typing import Callable, Tuple, Union

import torch

from ignite.contrib.metrics.regression._base import _BaseRegression
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class CanberraMetric(_BaseRegression):
    r"""Calculates the Canberra Metric.

    .. math::
        \text{CM} = \sum_{j=1}^n\frac{|A_j - P_j|}{|A_j| + |P_j|}

    where, :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    More details can be found in `Botchkarev 2018`_ or `scikit-learn distance metrics`_

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)`.

    .. _scikit-learn distance metrics:
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html

    .. versionchanged:: 0.4.3

        - Fixed implementation: ``abs`` in denominator.
        - Works with DDP.
    """

    def __init__(
        self, output_transform: Callable = lambda x: x, device: Union[str, torch.device] = torch.device("cpu")
    ) -> None:
        super(CanberraMetric, self).__init__(output_transform, device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_errors = torch.tensor(0.0, device=self._device)

    def _update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        y_pred, y = output
        errors = torch.abs(y - y_pred) / (torch.abs(y_pred) + torch.abs(y))
        self._sum_of_errors += torch.sum(errors).to(self._device)

    @sync_all_reduce("_sum_of_errors")
    def compute(self) -> float:
        return self._sum_of_errors.item()
