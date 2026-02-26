from typing import Callable

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["NDCG"]


class NDCG(Metric):
    r"""Calculates the Normalized Discounted Cumulative Gain (NDCG) at `k` for Recommendation Systems.

    NDCG measures the quality of ranking by considering both the relevance of items and their
    positions in the ranked list. It compares the achieved DCG against the ideal DCG (IDCG)
    obtained by sorting items by their true relevance.

    .. math::
        \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}

    where:

    .. math::
        \text{DCG}@K = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}

    and :math:`\text{rel}_i` is the relevance score of the item at position :math:`i` in the
    ranked list (1-indexed).

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - ``y_pred`` is expected to be raw logits or probability score for each item in the catalog.
    - ``y`` is expected to contain relevance scores (can be binary or graded).
    - ``y_pred`` and ``y`` are only allowed shape :math:`(batch, num\_items)`.
    - returns a list of NDCG ordered by the sorted values of ``top_k``.

    Args:
        top_k: a list of sorted positive integers that specifies `k` for calculating NDCG@top-k.
        ignore_zero_hits: if True, users with no relevant items (ground truth tensor being all zeros)
            are ignored in computation of NDCG. If set False, such users are counted with NDCG of 0.
            By default, True.
        relevance_threshold: minimum label value to be considered relevant. Defaults to ``1``,
            which handles standard binary labels and graded relevance scales (e.g. TREC-style
            0-4) by treating any label >= 1 as relevant. Items below this threshold contribute
            0 to DCG/IDCG calculations.
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

            metric = NDCG(top_k=[1, 2, 3, 4])
            metric.attach(default_evaluator, "ndcg")
            y_pred=torch.Tensor([
                [4.0, 2.0, 3.0, 1.0],
                [1.0, 2.0, 3.0, 4.0]
            ])
            y_true=torch.Tensor([
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0]
            ])
            state = default_evaluator.run([(y_pred, y_true)])
            print(state.metrics["ndcg"])

        .. testoutput:: 1

            [0.0, 0.63..., 0.63..., 0.63...]

        ignore_zero_hits=False case

        .. testcode:: 2

            metric = NDCG(top_k=[1, 2, 3, 4], ignore_zero_hits=False)
            metric.attach(default_evaluator, "ndcg")
            y_pred=torch.Tensor([
                [4.0, 2.0, 3.0, 1.0],
                [1.0, 2.0, 3.0, 4.0]
            ])
            y_true=torch.Tensor([
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0]
            ])
            state = default_evaluator.run([(y_pred, y_true)])
            print(state.metrics["ndcg"])

        .. testoutput:: 2

            [0.0, 0.31..., 0.31..., 0.31...]

    .. versionadded:: 0.6.0
    """

    required_output_keys = ("y_pred", "y")
    _state_dict_all_req_keys = ("_sum_ndcg_per_k", "_num_examples")

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
        super(NDCG, self).__init__(output_transform, device=device, skip_unrolling=skip_unrolling)

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_ndcg_per_k = torch.zeros(len(self.top_k), device=self._device)
        self._num_examples = 0

    def _compute_dcg(self, relevance_scores: torch.Tensor, k: int) -> torch.Tensor:
        """Compute DCG@k for a batch of relevance scores.
        
        Args:
            relevance_scores: Tensor of shape (batch, num_items) with relevance scores at ranked positions
            k: Number of positions to consider
            
        Returns:
            DCG scores of shape (batch,)
        """
        # Handle case where k > actual number of items
        actual_k = min(k, relevance_scores.shape[1])
        
        # Create position weights: 1/log2(position + 1) for position in [1, actual_k]
        # Positions are 1-indexed in the DCG formula
        positions = torch.arange(1, actual_k + 1, dtype=torch.float32, device=relevance_scores.device)
        discounts = 1.0 / torch.log2(positions + 1)  # log2(i+1) for i in [1, actual_k]
        
        # Compute gains: 2^rel - 1
        gains = torch.pow(2.0, relevance_scores[:, :actual_k]) - 1.0
        
        # DCG = sum of (gain / discount)
        dcg = (gains * discounts).sum(dim=-1)
        return dcg

    @reinit__is_reduced
    def update(self, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        if len(output) != 2:
            raise ValueError(f"output should be in format `(y_pred,y)` but got tuple of {len(output)} tensors.")

        y_pred, y = output
        if y_pred.shape != y.shape:
            raise ValueError(f"y_pred and y must be in the same shape, got {y_pred.shape} != {y.shape}.")

        # Filter out examples with no relevant items if ignore_zero_hits is True
        if self.ignore_zero_hits:
            valid_mask = torch.any(y >= self.relevance_threshold, dim=-1)
            y_pred = y_pred[valid_mask]
            y = y[valid_mask]

        if y.shape[0] == 0:
            return

        # Zero out items below relevance threshold for DCG computation
        y_for_dcg = torch.where(y >= self.relevance_threshold, y, torch.zeros_like(y))

        max_k = self.top_k[-1]

        # Get ranked indices based on predictions (stable=True for deterministic tie-breaking)
        ranked_indices = torch.argsort(y_pred, dim=-1, descending=True, stable=True)[:, :max_k]
        
        # Get relevance scores in the predicted ranking order
        ranked_relevance = torch.gather(y_for_dcg, dim=-1, index=ranked_indices)

        # Compute ideal ranking by sorting true relevance scores
        ideal_relevance = torch.sort(y_for_dcg, dim=-1, descending=True, stable=True)[0][:, :max_k]

        for i, k in enumerate(self.top_k):
            # Compute DCG@k and IDCG@k
            dcg_k = self._compute_dcg(ranked_relevance, k)
            idcg_k = self._compute_dcg(ideal_relevance, k)
            
            # NDCG = DCG / IDCG, handle division by zero (when IDCG = 0, NDCG = 0)
            ndcg_k = torch.where(
                idcg_k > 0,
                dcg_k / idcg_k,
                torch.zeros_like(dcg_k)
            )
            
            self._sum_ndcg_per_k[i] += ndcg_k.sum().to(self._device)

        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_ndcg_per_k", "_num_examples")
    def compute(self) -> list[float]:
        if self._num_examples == 0:
            raise NotComputableError("NDCG must have at least one example.")

        ndcg_scores = (self._sum_ndcg_per_k / self._num_examples).tolist()
        return ndcg_scores
