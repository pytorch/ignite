from typing import Callable

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["HitRate"]


class HitRate(Metric):
    """
    Calculates the Hit Rate at K for recommendation systems.

    The Hit Rate measures the fraction of users for which the model was able
    to predict atleast one correct recommendation.
    `Hit` for each user is either 0 or 1, irrespective of how many correct recommendations
    our model was able to predict for that user

    Args:
        top_k: a list of integers that specifies `k` for calculating hitrate@top-k.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric.
            The output is expected to be a tuple `(prediction, target)`
            where `prediction` and `target` are tensors
            of shape ``(batch_size, num_items)``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
        skip_unrolling: specifies whether input should be unrolled or not before being
            processed. Should be true for multi-output models.

    Attributes:
        required_output_keys: dictionary defines required keys to be found in ``engine.state.output`` if the
            latter is a dictionary. Default, ``("y_pred", "y")``.

    """

    required_output_keys = ("y_pred", "y")
    _state_dict_all_req_keys = ("_hits_per_k", "_num_examples")

    def __init__(
        self,
        top_k: list[int],
        output_transform: Callable = lambda x: x,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        self.top_k = sorted(top_k)
        super(HitRate, self).__init__(output_transform, device=device, skip_unrolling=skip_unrolling)

    @reinit__is_reduced
    def reset(self) -> None:
        self._hits_per_k = torch.zeros(len(self.top_k), device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        if len(output[0]) != 2:
            raise ValueError(f"data should be in format `(y_pred,y)` but got {len(output)} tensors.")
        y_pred, y = output

        max_k = self.top_k[-1]
        _, indices = torch.topk(y_pred, k=max_k, dim=-1)

        hits_at_max_k = torch.gather(y, dim=-1, index=indices)

        for i, k in enumerate(self.top_k):
            hit_mask = torch.any(hits_at_max_k[:, :k] > 0, dim=-1)
            self._hits_per_k[i] += torch.sum(hit_mask).to(self._device)

        self._num_examples += y.shape[0]

    @sync_all_reduce("_hits_per_k", "_num_examples")
    def compute(self) -> dict[str, float]:
        if self._num_examples == 0:
            raise NotComputableError("HitRate must have at least one example.")

        rates = (self._hits_per_k / self._num_examples).tolist()
        return {k: rate for k, rate in zip(self.top_k, rates)}
