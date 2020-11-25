from typing import Callable, Tuple, Union

import torch

from ignite.contrib.metrics.regression._base import _BaseRegression
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class ManhattanDistance(_BaseRegression):
    r"""
    Calculates the Manhattan Distance:

    :math:`\text{MD} = \sum_{j=1}^n |A_j - P_j|`,

    where :math:`A_j` is the ground truth and :math:`P_j` is the predicted value.

    More details can be found in `scikit-learn distance metrics`__.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    - `y` and `y_pred` must be of same shape `(N, )` or `(N, 1)`.

    __ https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html

    """

    def __init__(
        self, output_transform: Callable = lambda x: x, device: Union[str, torch.device] = torch.device("cpu")
    ):
        super(ManhattanDistance, self).__init__(output_transform, device)

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
