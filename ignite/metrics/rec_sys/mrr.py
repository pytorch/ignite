from collections.abc import Callable

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["MeanReciprocalRank"]


class MeanReciprocalRank(Metric):
    r"""Calculates Mean Reciprocal Rank (MRR) at k for Recommendation Systems.

    The Mean Reciprocal Rank measures the average of the reciprocal ranks of
    the first relevant item in the predicted ranking for each user.
    If no relevant item is found in the top-k predictions, the reciprocal rank for that user is 0.

    Math: MRR@K = (1 / N) * sum(1 / rank_i) for i = 1 to N

    - ``update`` must receive output of the form ``(y_pred, y)``.
    - ``y_pred`` is expected to be raw logits or probability score for each item in the catalog.
    - ``y`` is expected to be binary (only 0s and 1s) values where `1` indicates relevant item.
    - ``y_pred`` and ``y`` are only allowed shape :math:`(batch, num_items)`.
    - returns a list of MRR ordered by the sorted values of ``top_k``.

    Args:
        top_k: a single positive integer or a list of positive integers that specifies `k` for
            calculating MRR@top-k. If a single int is provided, it will be wrapped in a list.
            Default is 1.
        ignore_zero_hits: if True, users with no relevant items (ground truth tensor being all zeros)
            are ignored in computation of MRR. if set False, such users are counted with a reciprocal
            rank of 0. By default, True.
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
            processed. Should be true for multi-output models.

    Examples:
        .. testcode:: 1

            metric = MeanReciprocalRank(top_k=[1, 2, 3, 4])
            metric.attach(default_evaluator, "mrr")
            y_pred = torch.Tensor([
                [4.0, 2.0, 3.0, 1.0],
                [1.0, 2.0, 3.0, 4.0]
            ])
            y_true = torch.Tensor([
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0]
            ])
            state = default_evaluator.run([(y_pred, y_true)])
            print(state.metrics["mrr"])

        .. testoutput:: 1

            [0.0, 0.5, 0.5, 0.5]

        ignore_zero_hits=False case
        
        .. testcode:: 2

            metric = MeanReciprocalRank(top_k=[1, 2, 3, 4], ignore_zero_hits=False)
            metric.attach(default_evaluator, "mrr")
            y_pred = torch.Tensor([
                [4.0, 2.0, 3.0, 1.0],
                [1.0, 2.0, 3.0, 4.0]
            ])
            y_true = torch.Tensor([
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0]
            ])
            state = default_evaluator.run([(y_pred, y_true)])
            print(state.metrics["mrr"])

        .. testoutput:: 2

            [0.0, 0.25, 0.25, 0.25]
            
        int top_k case

        .. testcode:: 3

            metric = MeanReciprocalRank(top_k=2)
            metric.attach(default_evaluator, "mrr")
            y_pred = torch.Tensor([
                [4.0, 2.0, 3.0, 1.0],
            ])
            y_true = torch.Tensor([
                [0.0, 0.0, 1.0, 0.0],
            ])
            state = default_evaluator.run([(y_pred, y_true)])
            print(state.metrics["mrr"])

        .. testoutput:: 3

            [0.0]
    """

    required_output_keys = ("y_pred", "y")
    _state_dict_all_req_keys = ("_sum_mrr_per_k", "_num_examples")


    def __init__(
        self,
        top_k: list[int] | int = 1,
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
        self._sum_mrr_per_k = torch.zeros(len(self.top_k), device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self,output: tuple[torch.Tensor, torch.Tensor]) -> None:
        if len(output) != 2:
            raise ValueError(f"output should be in format `(y_pred,y)` but got tuple of {len(output)} tensors.")

        y_pred, y = output

        if y_pred.shape != y.shape:
            raise ValueError(f"y_pred and y must be in the same shape, got {y_pred.shape} != {y.shape}.")

        if self.ignore_zero_hits:
            # Remove users with no relevant items, like [0,0,0,0]
            valid_mask = torch.any(y > 0, dim=-1)
            y_pred = y_pred[valid_mask]
            y = y[valid_mask]

        if y.shape[0] == 0:
            return

        max_k = self.top_k[-1]
        # indices of top-ranked preds
        _, indices = torch.topk(y_pred, k=max_k, dim=-1)


        # Rearrange labels according to predicted ranking; example: [0,1,0,1]
        # ranked_relevance = [0,1,1,0]
        ranked_relevance = torch.gather(y, dim=-1, index=indices)

        for i, k in enumerate(self.top_k):

            # Only consider top-k recommendations
            relevance_k = ranked_relevance[:, :k]

            # Determine whether user has
            # at least one relevant item in top-k.
            has_hit = torch.any( relevance_k > 0, dim=-1)

            # Find position of first relevant item. Example: [0,0,1]
            # argmax -> 2; rank -> 3
            # edge case: argmax([0,0,0]) returns 0, but corrected using has_hit.
            first_pos = (torch.argmax((relevance_k > 0).int(),dim=-1) + 1)

            # Reciprocal rank; users with no hit receive 0
            rr = torch.where(has_hit, 1.0 / first_pos.float(), torch.zeros_like(first_pos, dtype=torch.float32))

            # Add batch contribution
            self._sum_mrr_per_k[i] += rr.sum().to(self._device)

        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_mrr_per_k", "_num_examples")
    def compute(self) -> list[float]:
        if self._num_examples == 0:
            raise NotComputableError("MeanReciprocalRank must have at least one example.")

        # avg reciprocal rank over all users
        rates = (self._sum_mrr_per_k/ self._num_examples).tolist()

        return rates
