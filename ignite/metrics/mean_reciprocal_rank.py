from typing import List, Callable, Union, Sequence

import torch

from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["MeanReciprocalRank"]

class MeanReciprocalRank(Metric):
    r"""Calculate `the mean reciprocal rank (MRR) <https://en.wikipedia.org/wiki/Mean_reciprocal_rank>`_.

    .. math:: \text{MRR} = \frac{1}{\lvert Q \rvert} \sum{i=1}^(\lvert Q \rvert) \frac{1}{rank_{i}}

    where :math:`rank_{i}` refers to the rank position of the first relevant document for the i-th query.

    Args:
        k: the k in “top-k”.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.

        Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in the format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``. If not, ``output_tranform`` can be added
        to the metric to transform the output into the form expected by the metric.

        ``y_pred`` and ``y`` should have the same shape.

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            metric = MeanReciprocalRank()
            metric.attach(default_evaluator, 'mrr')
            preds = torch.tensor([
                [1, 2, 4, 1],
                [2, 3, 1, 5],
                [1, 3, 5, 1],
                [1, 5, 1 ,11]
            ])
            target = preds * 0.75
            state = default_evaluator.run([[preds, target]])
            print(state.metrics['mrr'])

        .. testoutput::

 
    """

    def __init__(
        self,
        k: int = 5,
        output_transform: Callable = lambda x: torch.mean(x, 0),
        device: Union[str, torch.device] = torch.device("cpu")
    ):
        super(MeanReciprocalRank, self).__init__(output_transform=output_transform, device=device)
        self._k = k

    @reinit__is_reduced
    def reset(self):
        self._relevance = torch.empty(0)

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()
        _, topk_idx = y_pred.topk(self._k, dim=-1)
        relevance = y.take_along_dim(topk_idx, dim=-1)
        self._relevance = torch.cat([self._relevance, relevance], dim=-1)
        
    @sync_all_reduce("_sum", "_num_examples")
    def compute(self) -> float:
        first_relevant_positions = self._relevance.argmax(dim=-1) + 1
        valid_mask = (self._relevance.sum(dim=-1) > 0)

        return valid_mask/first_relevant_positions
