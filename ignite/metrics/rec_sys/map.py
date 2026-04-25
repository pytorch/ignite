from collections.abc import Callable

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["MAP"]


class MAP(Metric):
    r"""Calculates the Mean Average Precision (MAP) at `k` for Recommendation Systems.

    MAP measures the mean of Average Precision (AP) across all users. AP for a
    single user is the average of precision values computed at every position
    where a relevant item appears in the ranked top-k list, divided by the
    total number of relevant items for that user (clipped at ``k``).

    .. math::
        \text{AP}@K_i = \frac{1}{\min(R_i, K)}
        \sum_{j=1}^{K} \text{Precision}@j \cdot \mathbb{1}(\text{rel}_{i,j})

    .. math::
        \text{MAP}@K = \frac{1}{N} \sum_{i=1}^{N} \text{AP}@K_i

    where :math:`R_i` is the number of relevant items for user :math:`i`,
    :math:`\text{rel}_{i,j}` is 1 if the item at rank :math:`j` is relevant
    and 0 otherwise, and :math:`\text{Precision}@j` is the proportion of
    relevant items in the top :math:`j` ranked predictions.

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - ``y_pred`` is expected to be raw logits or probability scores for each item
      in the catalog.
    - ``y`` is expected to be binary (only 0s and 1s) values where ``1`` indicates
      a relevant item.
    - ``y_pred`` and ``y`` are only allowed shape :math:`(batch, num\_items)`.
    - returns a list of MAP values ordered by the sorted values of ``top_k``.

    Args:
        top_k: a single positive integer or a list of positive integers that specifies
            ``k`` for calculating MAP@top-k. If a single int is provided, it will be
            wrapped in a list. Default is 10.
        ignore_zero_hits: if True, users with no relevant items (ground truth tensor
            being all zeros) are ignored in computation of MAP. If set False, such
            users are counted with an Average Precision of 0. By default, True.
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

            metric = MAP(top_k=[1, 2, 3, 4])
            metric.attach(default_evaluator, "map")
            y_pred = torch.Tensor([
                [4.0, 2.0, 3.0, 1.0],
                [1.0, 2.0, 3.0, 4.0],
            ])
            y_true = torch.Tensor([
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            state = default_evaluator.run([(y_pred, y_true)])
            print(state.metrics["map"])

    .. versionadded:: 0.6.0
    """

    required_output_keys = ("y_pred", "y")
    _state_dict_all_req_keys = ("_sum_ap_per_k", "_num_examples")

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
        self._sum_ap_per_k = torch.zeros(len(self.top_k), device=self._device)
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
        _, indices = torch.topk(y_pred, k=max_k, dim=-1)
        ranked_relevance = torch.gather(y, dim=-1, index=indices).to(torch.float32)

        # Total number of relevant items per user across the catalog (used as
        # the AP denominator, clipped at k below).
        total_relevant = (y > 0).to(torch.float32).sum(dim=-1)

        for i, k in enumerate(self.top_k):
            top_k_relevance = ranked_relevance[:, :k]
            # Cumulative number of relevant items at each rank up to k.
            cumulative_hits = torch.cumsum(top_k_relevance, dim=-1)
            positions = torch.arange(1, k + 1, dtype=torch.float32, device=top_k_relevance.device)
            # Precision@j evaluated at every rank j in [1, k].
            precision_at_j = cumulative_hits / positions
            # Sum precision values only at positions where the item is relevant.
            sum_precision = (precision_at_j * top_k_relevance).sum(dim=-1)

            denom = torch.clamp(total_relevant, max=float(k))
            ap_k = torch.where(
                denom > 0,
                sum_precision / denom,
                torch.zeros_like(sum_precision),
            )
            self._sum_ap_per_k[i] += ap_k.sum().to(self._device)

        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_ap_per_k", "_num_examples")
    def compute(self) -> list[float]:
        if self._num_examples == 0:
            raise NotComputableError("MAP must have at least one example.")

        rates = (self._sum_ap_per_k / self._num_examples).tolist()
        return rates
