from typing import Callable, Sequence, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["TopKCategoricalAccuracy"]


class TopKCategoricalAccuracy(Metric):
    """
    Calculates the top-k categorical accuracy.

    - ``update`` must receive output of the form ``(y_pred, y)``.

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
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-ouput as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in the format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``. If not, ``output_tranform`` can be added
        to the metric to transform the output into the form expected by the metric.

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            def process_function(engine, batch):
                y_pred, y = batch
                return y_pred, y

            def one_hot_to_binary_output_transform(output):
                y_pred, y = output
                y = torch.argmax(y, dim=1)  # one-hot vector to label index vector
                return y_pred, y

            engine = Engine(process_function)
            metric = TopKCategoricalAccuracy(
                k=2, output_transform=one_hot_to_binary_output_transform)
            metric.attach(engine, 'top_k_accuracy')

            preds = torch.tensor([
                [0.7, 0.2, 0.05, 0.05],     # 1 is in the top 2
                [0.2, 0.3, 0.4, 0.1],       # 0 is not in the top 2
                [0.4, 0.4, 0.1, 0.1],       # 0 is in the top 2
                [0.7, 0.05, 0.2, 0.05]      # 2 is in the top 2
            ])
            target = torch.tensor([         # targets as one-hot vectors
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0]
            ])

            state = engine.run([[preds, target]])
            print(state.metrics['top_k_accuracy'])

        .. testoutput::

            0.75

    .. versionchanged:: 0.5.1
        ``skip_unrolling`` argument is added.
    """

    _state_dict_all_req_keys = ("_num_correct", "_num_examples")

    def __init__(
        self,
        k: int = 5,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        skip_unrolling: bool = False,
    ) -> None:
        super(TopKCategoricalAccuracy, self).__init__(output_transform, device=device, skip_unrolling=skip_unrolling)
        self._k = k

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_correct = torch.tensor(0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()
        sorted_indices = torch.topk(y_pred, self._k, dim=1)[1]
        expanded_y = y.view(-1, 1).expand(-1, self._k)
        correct = torch.sum(torch.eq(sorted_indices, expanded_y), dim=1)

        self._num_correct += torch.sum(correct).to(self._device)
        self._num_examples += correct.shape[0]

    @sync_all_reduce("_num_correct", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError(
                "TopKCategoricalAccuracy must have at least one example before it can be computed."
            )
        return self._num_correct.item() / self._num_examples
