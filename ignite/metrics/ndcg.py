from typing import Callable, Sequence, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric

__all__ = ["NGCD"]


class NGCD(Metric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        k=None,
        log_base=2,
    ):
        super(NGCD, self).__init__(output_transform=output_transform, device=device)

        self.log_base = log_base
        self.k = k
        self.num_examples = 0
        self.ngcd = torch.tensor(0, device=self._device, dtype=torch.float32)

    def _dcg_sample_scores(self, y_true, y_score, k=None, log_base=2) -> torch.Tensor:

        discount = torch.div(torch.log(torch.arange(y_true.shape[1]) + 2), torch.log(torch.tensor(log_base))).pow(-1)

        if k is not None:
            discount[k:] = 0

        ranking = torch.argsort(y_score, descending=True)

        ranked = torch.zeros(y_score.shape, dtype=y_score.dtype)
        ranked = ranked.scatter_(1, ranking, y_true)

        discounted_gains = torch.mm(ranked, discount.reshape(-1, 1))

        return discounted_gains

    def _ndcg_sample_scores(self, y_true, y_score, k=None, log_base=2) -> torch.Tensor:

        gain = self._dcg_sample_scores(y_true, y_score, k, log_base=log_base)

        normalizing_gain = self._dcg_sample_scores(y_true, y_true, k, log_base=log_base)

        all_irrelevant = normalizing_gain == 0

        normalized_gain = torch.div(gain[~all_irrelevant], normalizing_gain[~all_irrelevant])

        return normalized_gain

    def reset(self) -> None:

        self.num_examples = 0
        self.ngcd = torch.tensor(0, device=self._device)

    def update(self, output: Sequence[torch.Tensor]) -> None:

        y_pred, y = output[0], output[1]

        gain = self._ndcg_sample_scores(y, y_pred, k=self.k)

        self.ngcd += torch.sum(gain)

        self.num_examples += y_pred.shape[0]

    def compute(self) -> float:
        if self.num_examples == 0:
            raise NotComputableError("NGCD must have at least one example before it can be computed.")

        return self.ngcd / self.num_examples
