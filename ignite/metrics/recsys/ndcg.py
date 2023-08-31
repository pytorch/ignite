from typing import Callable, Optional, Sequence, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["NDCG"]


def _tie_averaged_dcg(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    discount_cumsum: torch.Tensor,
    device: Union[str, torch.device] = torch.device("cpu"),
) -> torch.Tensor:

    _, inv, counts = torch.unique(-y_pred, return_inverse=True, return_counts=True)
    ranked = torch.zeros(counts.shape[0]).to(device)
    ranked.index_put_([inv], y_true, accumulate=True)
    ranked /= counts
    groups = torch.cumsum(counts, dim=-1) - 1
    discount_sums = torch.empty(counts.shape[0]).to(device)
    discount_sums[0] = discount_cumsum[groups[0]]
    discount_sums[1:] = torch.diff(discount_cumsum[groups])

    return torch.sum(torch.mul(ranked, discount_sums))


def _dcg_sample_scores(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    k: Optional[int] = None,
    log_base: Union[int, float] = 2,
    ignore_ties: bool = False,
    device: Union[str, torch.device] = torch.device("cpu"),
) -> torch.Tensor:

    discount = torch.log(torch.tensor(log_base)) / torch.log(torch.arange(y_true.shape[1]) + 2)
    discount = discount.to(device)

    if k is not None:
        discount[k:] = 0.0

    if ignore_ties:
        ranking = torch.argsort(y_pred, descending=True)
        ranked = y_true[torch.arange(ranking.shape[0]).reshape(-1, 1), ranking].to(device)
        discounted_gains = torch.mm(ranked, discount.reshape(-1, 1))

    else:
        discount_cumsum = torch.cumsum(discount, dim=-1)
        discounted_gains = torch.tensor(
            [_tie_averaged_dcg(y_p, y_t, discount_cumsum, device) for y_p, y_t in zip(y_pred, y_true)], device=device
        )

    return discounted_gains


def _ndcg_sample_scores(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    k: Optional[int] = None,
    log_base: Union[int, float] = 2,
    ignore_ties: bool = False,
) -> torch.Tensor:

    device = y_true.device
    gain = _dcg_sample_scores(y_pred, y_true, k=k, log_base=log_base, ignore_ties=ignore_ties, device=device)
    if not ignore_ties:
        gain = gain.unsqueeze(dim=-1)
    normalizing_gain = _dcg_sample_scores(y_true, y_true, k=k, log_base=log_base, ignore_ties=True, device=device)
    all_relevant = normalizing_gain != 0
    normalized_gain = gain[all_relevant] / normalizing_gain[all_relevant]
    return normalized_gain


class NDCG(Metric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        k: Optional[int] = None,
        log_base: Union[int, float] = 2,
        exponential: bool = False,
        ignore_ties: bool = False,
    ):

        if log_base == 1 or log_base <= 0:
            raise ValueError(f"Argument log_base should positive and not equal one,but got {log_base}")
        self.log_base = log_base
        self.k = k
        self.exponential = exponential
        self.ignore_ties = ignore_ties
        super(NDCG, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:

        self.num_examples = 0
        self.ndcg = torch.tensor(0.0, device=self._device)

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:

        y_pred, y_true = output[0].detach(), output[1].detach()

        y_pred = y_pred.to(torch.float32).to(self._device)
        y_true = y_true.to(torch.float32).to(self._device)

        if self.exponential:
            y_true = 2 ** y_true - 1

        gain = _ndcg_sample_scores(y_pred, y_true, k=self.k, log_base=self.log_base, ignore_ties=self.ignore_ties)
        self.ndcg += torch.sum(gain)
        self.num_examples += y_pred.shape[0]

    @sync_all_reduce("ndcg", "num_examples")
    def compute(self) -> float:
        if self.num_examples == 0:
            raise NotComputableError("NGCD must have at least one example before it can be computed.")

        return (self.ndcg / self.num_examples).item()