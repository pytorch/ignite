from typing import Callable

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["MRR"]


class MRR(Metric):
    r"""Calculates the Mean Reciprocal Rank (MRR) at `k` for Recommendation Systems.

    MRR measures the average of the reciprocal of the rank of the first relevant item
    in the predicted list. It is widely used in retrieval systems, recommendation systems,
    and RAG pipelines.

    .. math:: \text{MRR}@K = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}

    where :math:`\text{rank}_i` is the rank (1-indexed) of the first relevant item
    in the top-K predictions for user :math:`i`. If no relevant item is found in the
    top-K, the reciprocal rank for that user is 0.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - ``y_pred`` is expected to be raw logits or probability score for each item in the catalog.
    - ``y`` is expected to be binary (only 0s and 1s) values where `1` indicates relevant item.
      Graded relevance labels are also supported via ``relevance_threshold``.
    - ``y_pred`` and ``y`` are only allowed shape :math:`(batch, num\_items)`.
    - returns a list of MRR ordered by the sorted values of ``top_k``.

    Args:
        top_k: a list of sorted positive integers that specifies `k` for calculating MRR@top-k.
        ignore_zero_hits: if True, users with no relevant items (ground truth tensor being all zeros)
            are ignored in computation of MRR. If set False, such users are counted as having
            reciprocal rank of 0. By default, True.
        relevance_threshold: minimum label value to be considered relevant. Defaults to ``1``,
            which handles standard binary labels and graded relevance scales (e.g. TREC-style
            0-4) by treating any label >= 1 as relevant.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric.
            The output is expected to be a tuple `(prediction, target)`
            where `prediction` and `target` are tensors
            of shape ``(batch, num_items)``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
        skip_unrolling: specifies whether input should be unrolled or not before being
            processed. Should be true for multi-output models..

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in the format of
        ``(y_pred, y)``. If not, ``output_tranform`` can be added
        to the metric to transform the output into the form expected by the metric.

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        ignore_zero_hits=True case

        .. testcode:: 1

            metric = MRR(top_k=[1, 2, 3, 4])
            metric.attach(default_evaluator,"mrr")
            y_pred=torch.Tensor([
                [4.0, 2.0, 3.0, 1.0],
                [1.0, 2.0, 3.0, 4.0]
            ])
            y_true=torch.Tensor([
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0]
            ])
            state = default_evaluator.run([(y_pred, y_true)])
            print(state.metrics["mrr"])

        .. testoutput:: 1

            [0.0, 0.5, 0.5, 0.5]

        ignore_zero_hits=False case

        .. testcode:: 2

            metric = MRR(top_k=[1, 2, 3, 4], ignore_zero_hits=False)
            metric.attach(default_evaluator,"mrr")
            y_pred=torch.Tensor([
                [4.0, 2.0, 3.0, 1.0],
                [1.0, 2.0, 3.0, 4.0]
            ])
            y_true=torch.Tensor([
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0]
            ])
            state = default_evaluator.run([(y_pred, y_true)])
            print(state.metrics["mrr"])

        .. testoutput:: 2

            [0.0, 0.25, 0.25, 0.25]

    .. versionadded:: 0.6.0
    """

    required_output_keys = ("y_pred", "y")
    _state_dict_all_req_keys = ("_sum_reciprocal_ranks_per_k", "_num_examples")

    def __init__(
        self,
        top_k: list[int],
        ignore_zero_hits: bool = True,
        relevance_threshold: float = 1.0,
        output_transform: Callable = lambda x: x,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        if any(k <= 0 for k in top_k):
            raise ValueError(" top_k must be list of positive integers only.")

        self.top_k = sorted(top_k)
        self.ignore_zero_hits = ignore_zero_hits
        self.relevance_threshold = relevance_threshold
        super(MRR, self).__init__(output_transform, device=device, skip_unrolling=skip_unrolling)

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_reciprocal_ranks_per_k = torch.zeros(len(self.top_k), device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        if len(output) != 2:
            raise ValueError(f"output should be in format `(y_pred,y)` but got tuple of {len(output)} tensors.")

        y_pred, y = output
        if y_pred.shape != y.shape:
            raise ValueError(f"y_pred and y must be in the same shape, got {y_pred.shape} != {y.shape}.")

        if self.ignore_zero_hits:
            valid_mask = torch.any(y >= self.relevance_threshold, dim=-1)
            y_pred = y_pred[valid_mask]
            y = y[valid_mask]

        if y.shape[0] == 0:
            return

        max_k = self.top_k[-1]

        # stable=True ensures deterministic tie-breaking, consistent with
        # reference libraries such as ranx.
        ranked_indices = torch.argsort(y_pred, dim=-1, descending=True, stable=True)[:, :max_k]
        ranked_labels = torch.gather(y, dim=-1, index=ranked_indices)

        for i, k in enumerate(self.top_k):
            top_k_labels = ranked_labels[:, :k]
            relevant_mask = top_k_labels >= self.relevance_threshold

            has_hit = relevant_mask.any(dim=-1)

            # argmax on int tensor returns 0-based position of first True
            first_hit_pos = relevant_mask.int().argmax(dim=-1)

            reciprocal_rank = torch.where(
                has_hit,
                1.0 / (first_hit_pos.float() + 1.0),
                torch.zeros_like(first_hit_pos, dtype=torch.float),
            )

            self._sum_reciprocal_ranks_per_k[i] += reciprocal_rank.sum().to(self._device)

        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_reciprocal_ranks_per_k", "_num_examples")
    def compute(self) -> list[float]:
        if self._num_examples == 0:
            raise NotComputableError("MRR must have at least one example.")

        rates = (self._sum_reciprocal_ranks_per_k / self._num_examples).tolist()
        return rates
        