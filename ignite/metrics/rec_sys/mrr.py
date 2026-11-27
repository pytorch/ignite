from collections.abc import Callable, Sequence
import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["MRR"]


class MRR(Metric):
    def __init__(
        self,
        top_k: Sequence[int],
        ignore_zero_hits: bool = True,
        relevance_threshold: float = 1.0,
        output_transform: Callable = lambda x: x,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
    ) -> None:
        if not isinstance(top_k, Sequence) or len(top_k) == 0:
            raise ValueError(f"top_k must be a non-empty sequence of positive integers, got {top_k}.")
        if any(not isinstance(k, int) or k <= 0 for k in top_k):
            raise ValueError(f"top_k must contain only positive integers, got {top_k}.")

        self.top_k: list[int] = sorted(set(top_k))
        self.ignore_zero_hits = ignore_zero_hits
        self.relevance_threshold = relevance_threshold
        super().__init__(output_transform=output_transform, device=device, skip_unrolling=skip_unrolling)

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_rr = torch.zeros(len(self.top_k), dtype=torch.float64, device=self._device)
        self._num_examples = 0

    def _check_shape_and_type(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        if y_pred.ndim != 2 or y.ndim != 2:
            raise ValueError(
                f"y_pred and y must be 2D tensors of shape (batch_size, num_items), got {y_pred.shape} and {y.shape}."
            )
        if y_pred.shape != y.shape:
            raise ValueError(f"y_pred and y must have the same shape, got {y_pred.shape} and {y.shape}.")
        if not y_pred.dtype.is_floating_point:
            raise ValueError(f"y_pred must be a floating point tensor, got {y_pred.dtype}.")

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()
        self._check_shape_and_type(y_pred, y)

        y_pred = y_pred.to(self._device)
        y = y.to(self._device)

        relevant = y >= self.relevance_threshold

        if self.ignore_zero_hits:
            valid_users_mask = relevant.any(dim=1)
            if not valid_users_mask.any():
                return
            y_pred = y_pred[valid_users_mask]
            relevant = relevant[valid_users_mask]

        batch_size, num_items = y_pred.shape
        max_k = min(max(self.top_k), num_items)

        _, topk_indices = torch.topk(y_pred, k=max_k, dim=1, largest=True, sorted=True)
        hits_at_max_k = torch.gather(relevant, dim=1, index=topk_indices)

        ranks = torch.arange(1, max_k + 1, dtype=torch.float64, device=self._device).unsqueeze(0)
        reciprocal_ranks = torch.where(hits_at_max_k, 1.0 / ranks, torch.zeros_like(ranks))

        for idx, k in enumerate(self.top_k):
            cutoff = min(k, num_items)
            max_rr_per_user, _ = reciprocal_ranks[:, :cutoff].max(dim=1)
            self._sum_rr[idx] += max_rr_per_user.sum()

        self._num_examples += batch_size

    @sync_all_reduce("_sum_rr", "_num_examples")
    def compute(self) -> list[float]:
        if self._num_examples == 0:
            raise NotComputableError("MRR must have at least one valid example before it can be computed.")
        return (self._sum_rr / self._num_examples).tolist()
