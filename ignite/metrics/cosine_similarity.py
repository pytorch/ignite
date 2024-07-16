from typing import Callable, Sequence, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["CosineSimilarity"]


class CosineSimilarity(Metric):
    r"""Calculates the mean of the `cosine similarity <https://en.wikipedia.org/wiki/Cosine_similarity>`_.

    .. math::
       \text{cosine\_similarity} = \frac{1}{N} \sum_{i=1}^N
       \frac{x_i \cdot y_i}{\max ( \| x_i \|_2 \| y_i \|_2 , \epsilon)}

    where :math:`y_{i}` is the prediction tensor and :math:`x_{i}` is ground true tensor.

    - ``update`` must receive output of the form ``(y_pred, y)``.

    Args:
        eps: a small value to avoid division by zero. Default: 1e-8
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-ouput as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

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

            metric = CosineSimilarity()
            metric.attach(default_evaluator, 'cosine_similarity')
            preds = torch.tensor([
                [1, 2, 4, 1],
                [2, 3, 1, 5],
                [1, 3, 5, 1],
                [1, 5, 1 ,11]
            ]).float()
            target = torch.tensor([
                [1, 5, 1 ,11],
                [1, 3, 5, 1],
                [2, 3, 1, 5],
                [1, 2, 4, 1]
            ]).float()
            state = default_evaluator.run([[preds, target]])
            print(state.metrics['cosine_similarity'])

        .. testoutput::

            0.5080491304397583

    .. versionchanged:: 0.5.1
        ``skip_unrolling`` argument is added.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        super().__init__(output_transform, device, skip_unrolling=skip_unrolling)

        self.eps = eps

    _state_dict_all_req_keys = ("_sum_of_cos_similarities", "_num_examples")

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_cos_similarities = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred = output[0].flatten(start_dim=1).detach()
        y = output[1].flatten(start_dim=1).detach()
        cos_similarities = torch.nn.functional.cosine_similarity(y_pred, y, dim=1, eps=self.eps)
        self._sum_of_cos_similarities += torch.sum(cos_similarities).to(self._device)
        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_of_cos_similarities", "_num_examples")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("CosineSimilarity must have at least one example before it can be computed.")
        return self._sum_of_cos_similarities.item() / self._num_examples
