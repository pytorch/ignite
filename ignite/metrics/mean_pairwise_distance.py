from typing import Callable, Sequence, Union

import torch
from torch.nn.functional import pairwise_distance

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["MeanPairwiseDistance"]


class MeanPairwiseDistance(Metric):
    """Calculates the mean :class:`~torch.nn.PairwiseDistance`.
    Average of pairwise distances computed on provided batches.

    - ``update`` must receive output of the form ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.

    Args:
        p: the norm degree. Default: 2
        eps: Small value to avoid division by zero. Default: 1e-6
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

        .. testcode::

            metric = MeanPairwiseDistance(p=4)
            metric.attach(default_evaluator, 'mpd')
            preds = torch.Tensor([
                [1, 2, 4, 1],
                [2, 3, 1, 5],
                [1, 3, 5, 1],
                [1, 5, 1 ,11]
            ])
            target = preds * 0.75
            state = default_evaluator.run([[preds, target]])
            print(state.metrics['mpd'])

        .. testoutput::

            1.5955...
    """

    def __init__(
        self,
        p: int = 2,
        eps: float = 1e-6,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        super(MeanPairwiseDistance, self).__init__(output_transform, device=device)
        self._p = p
        self._eps = eps

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_distances = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()
        distances = pairwise_distance(y_pred, y, p=self._p, eps=self._eps)
        self._sum_of_distances += torch.sum(distances).to(self._device)
        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_of_distances", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError("MeanAbsoluteError must have at least one example before it can be computed.")
        return self._sum_of_distances.item() / self._num_examples
