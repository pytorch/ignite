from typing import Callable, Optional, Sequence, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["TopKCategoricalAccuracy"]


class TopKCategoricalAccuracy(Metric):
    """
    Calculates the top-k categorical accuracy.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
    """

    def __init__(
        self, k=5, output_transform: Callable = lambda x: x, device: Optional[Union[str, torch.device]] = None
    ):
        super(TopKCategoricalAccuracy, self).__init__(output_transform, device=device)
        self._k = k

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_correct = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence) -> None:
        y_pred, y = output
        sorted_indices = torch.topk(y_pred, self._k, dim=1)[1]
        expanded_y = y.view(-1, 1).expand(-1, self._k)
        correct = torch.sum(torch.eq(sorted_indices, expanded_y), dim=1)
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]

    @sync_all_reduce("_num_correct", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError(
                "TopKCategoricalAccuracy must have at" "least one example before it can be computed."
            )
        return self._num_correct / self._num_examples
