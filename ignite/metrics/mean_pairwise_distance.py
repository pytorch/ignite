from typing import Callable, Sequence, Union

import torch
from torch.nn.functional import pairwise_distance

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["MeanPairwiseDistance"]


class MeanPairwiseDistance(Metric):
    """
    Calculates the mean pairwise distance: average of pairwise distances computed on provided batches.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    """

    def __init__(
        self,
        p: int = 2,
        eps: float = 1e-6,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(MeanPairwiseDistance, self).__init__(output_transform, device=device)
        self._p = p
        self._eps = eps

    @reinit__is_reduced
    def reset(self):
        self._sum_of_distances = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()
        distances = pairwise_distance(y_pred, y, p=self._p, eps=self._eps)
        self._sum_of_distances += torch.sum(distances).to(self._device)
        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_of_distances", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError("MeanAbsoluteError must have at least one example before it can be computed.")
        return self._sum_of_distances.item() / self._num_examples
