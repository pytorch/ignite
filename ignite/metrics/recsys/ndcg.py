from typing import Callable, Optional, Sequence, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric

__all__ = ["NDCG"]


class NDCG(Metric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        k: Optional[int] = None,
        log_base: Union[int, float] = 2,
    ):

        self.log_base = log_base
        self.k = k
        super(NDCG, self).__init__(output_transform=output_transform, device=device)

    def _dcg_sample_scores(
        self, y_true: torch.Tensor, y_score: torch.Tensor, k: Optional[int] = None, log_base: Union[int, float] = 2
    ) -> torch.Tensor:

        discount = (
            torch.div(torch.log(torch.arange(y_true.shape[1]) + 2), torch.log(torch.tensor(log_base)))
            .pow(-1)
            .to(self._device)
        )
        if k is not None:
            discount[k:] = 0.0

        ranking = torch.argsort(y_score, descending=True)
        ranked = y_true[torch.arange(ranking.shape[0]).reshape(-1, 1), ranking].to(self._device)
        discounted_gains = torch.mm(ranked, discount.reshape(-1, 1))
        return discounted_gains

    def _ndcg_sample_scores(
        self, y_true: torch.Tensor, y_score: torch.Tensor, k: Optional[int] = None, log_base: Union[int, float] = 2
    ) -> torch.Tensor:

        gain = self._dcg_sample_scores(y_true, y_score, k, log_base=log_base)
        normalizing_gain = self._dcg_sample_scores(y_true, y_true, k, log_base=log_base)
        all_relevant = normalizing_gain != 0
        normalized_gain = torch.div(gain[all_relevant], normalizing_gain[all_relevant])
        return normalized_gain

    def reset(self) -> None:

        self.num_examples = 0
        self.ndcg = torch.tensor(0.0, device=self._device)

    def update(self, output: Sequence[torch.Tensor]) -> None:

        y, y_pred = output[0].detach(), output[1].detach()
        gain = self._ndcg_sample_scores(y, y_pred, k=self.k, log_base=self.log_base)
        self.ndcg = torch.add(self.ndcg, torch.sum(gain))
        self.num_examples += y_pred.shape[0]

    def compute(self) -> float:
        if self.num_examples == 0:
            raise NotComputableError("NGCD must have at least one example before it can be computed.")

        return (self.ndcg / self.num_examples).item()
