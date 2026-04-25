from collections.abc import Callable

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["MRR"]


class MRR(Metric):
    r"""Calculates the Mean Reciprocal Rank (MRR) at `k` for Recommendation Systems.

    MRR measures the average reciprocal rank of the first relevant item in the
    predicted ranking for each user. The reciprocal rank for a user is
    :math:`1/\text{rank}` where :math:`\text{rank}` is the position of the first
    relevant item in the ranked list (1-indexed). Users for which no relevant
    item appears in the top-k results contribute 0 to the score.

    .. math:: \text{MRR}@K = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}

    where :math:`\text{rank}_i \in \{1, 2, \ldots, K\}` is the rank of the first
    relevant item for user :math:`i` in the top-k predictions, and is treated as
    :math:`\infty` (yielding a reciprocal rank of 0) if no relevant item is in
    the top-k predictions.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - ``y_pred`` is expected to be raw logits or probability scores for each item
      in the catalog.
    - ``y`` is expected to be binary (only 0s and 1s) values where ``1`` indicates
      a relevant item.
    - ``y_pred`` and ``y`` are only allowed shape :math:`(batch, num\_items)`.
    - returns a list of MRR values ordered by the sorted values of ``top_k``.

    Args:
        top_k: a single positive integer or a list of positive integers that specifies
            ``k`` for calculating MRR@top-k. If a single int is provided, it will be
            wrapped in a list. Default is 10.
        ignore_zero_hits: if True, users with no relevant items (ground truth tensor
            being all zeros) are ignored in computation of MRR. If set False, such
            users are counted with a reciprocal rank of 0. By default, True.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into
            the form expected by the metric.
            The output is expected to be a tuple ``(prediction, target)`` where
            ``prediction`` and ``target`` are tensors of shape ``(batch, num_items)``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the
            ``update`` method is non-blocking. By default, CPU.
        skip_unrolling: specifies whether input should be unrolled or not before
            being processed. Should be true for multi-output models.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric
        instance to the engine. The output of the engine's ``process_function``
        needs to be in the format of ``(y_pred, y)``. If not, ``output_transform``
        can be added to the metric to transform the output into the form expected
        by the metric.

        For more information on how metric works with
        :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = MRR(top_k=[1, 2, 3, 4])
            metric.attach(default_evaluator, "mrr")
            y_pred = torch.Tensor([
                [4.0, 2.0, 3.0, 1.0],
                [1.0, 2.0, 3.0, 4.0],
            ])
            y_true = torch.Tensor([
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            state = default_evaluator.run([(y_pred, y_true)])
            print(state.metrics["mrr"])

        .. testoutput::

            [0.5, 0.75, 0.75, 0.75]

    .. versionadded:: 0.6.0
    """

    required_output_keys = ("y_pred", "y")
    _state_dict_all_req_keys = ("_sum_rr_per_k", "_num_examples")

    def __init__(
        self,
        top_k: list[int] | int = 10,
        ignore_zero_hits: bool = True,
        output_transform: Callable = lambda x: x,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        if not isinstance(top_k, (int, list)):
            raise ValueError("top_k must be either int or a list[int]")

        top_k = [top_k] if isinstance(top_k, int) else top_k

        if len(top_k) == 0:
            raise ValueError("top_k must have at least one positive value")
        if any(k <= 0 for k in top_k):
            raise ValueError("top_k must be list of positive integers only.")

        self.top_k = sorted(top_k)
        self.ignore_zero_hits = ignore_zero_hits
        super().__init__(output_transform, device=device, skip_unrolling=skip_unrolling)

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_rr_per_k = torch.zeros(len(self.top_k), device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        if len(output) != 2:
            raise ValueError(f"output should be in format `(y_pred,y)` but got tuple of {len(output)} tensors.")

        y_pred, y = output
        if y_pred.shape != y.shape:
            raise ValueError(f"y_pred and y must be in the same shape, got {y_pred.shape} != {y.shape}.")

        if self.ignore_zero_hits:
            valid_mask = torch.any(y > 0, dim=-1)
            y_pred = y_pred[valid_mask]
            y = y[valid_mask]

        if y.shape[0] == 0:
            return

        max_k = self.top_k[-1]
        # Get top-max_k predictions ordered by descending score.
        _, indices = torch.topk(y_pred, k=max_k, dim=-1)
        # Gather corresponding relevance labels in ranked order.
        ranked_relevance = torch.gather(y, dim=-1, index=indices)

        batch_size = y.shape[0]

        for i, k in enumerate(self.top_k):
            top_k_relevance = ranked_relevance[:, :k]
            # First-relevant position per user (1-indexed). For users with no
            # relevant item in top-k, fall back to a sentinel position whose
            # reciprocal rank evaluates to 0.
            has_hit = torch.any(top_k_relevance > 0, dim=-1)
            # argmax returns the index of the first max; for binary labels that
            # is the position of the first 1 when at least one exists.
            first_pos = torch.argmax(top_k_relevance.to(torch.long), dim=-1) + 1
            reciprocal_rank = torch.where(
                has_hit,
                1.0 / first_pos.to(torch.float32),
                torch.zeros(batch_size, device=top_k_relevance.device),
            )
            self._sum_rr_per_k[i] += reciprocal_rank.sum().to(self._device)

        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_rr_per_k", "_num_examples")
    def compute(self) -> list[float]:
        if self._num_examples == 0:
            raise NotComputableError("MRR must have at least one example.")

        rates = (self._sum_rr_per_k / self._num_examples).tolist()
        return rates
