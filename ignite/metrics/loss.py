from typing import Callable, cast, Dict, Sequence, Tuple, Union

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["Loss"]


class Loss(Metric):
    """
    Calculates the average loss according to the passed loss_fn.

    Args:
        loss_fn: a callable taking a prediction tensor, a target
            tensor, optionally other arguments, and returns the average loss
            over all observations in the batch.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric.
            This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            The output is expected to be a tuple `(prediction, target)` or
            (prediction, target, kwargs) where kwargs is a dictionary of extra
            keywords arguments. If extra keywords arguments are provided they are passed to `loss_fn`.
        batch_size: a callable taking a target tensor that returns the
            first dimension size (usually the batch size).
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
        skip_unrolling: specifies whether input should be unrolled or not before it is passed to to loss_fn.
            Should be true for multi-output model, for example, if ``y_pred`` contains multi-ouput as
            ``(y_pred_a, y_pred_b)``

    Attributes:
        required_output_keys: dictionary defines required keys to be found in ``engine.state.output`` if the
            latter is a dictionary. Default, ``("y_pred", "y", "criterion_kwargs")``. This is useful when the
            criterion function requires additional arguments, which can be passed using ``criterion_kwargs``.
            See an example below.

    Examples:
        Let's implement a Loss metric that requires ``x``, ``y_pred``, ``y`` and ``criterion_kwargs`` as input
        for ``criterion`` function. In the example below we show how to setup standard metric like Accuracy
        and the Loss metric using an ``evaluator`` created with
        :meth:`~ignite.engine.create_supervised_evaluator` method.

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            model = default_model
            criterion = nn.NLLLoss()
            metric = Loss(criterion)
            metric.attach(default_evaluator, 'loss')
            y_pred = torch.tensor([[0.1, 0.4, 0.5], [0.1, 0.7, 0.2]])
            y_true = torch.tensor([2, 2]).long()
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics['loss'])

        .. testoutput::

            -0.3499999...

    .. versionchanged:: 0.5.1
        ``skip_unrolling`` argument is added.
    """

    required_output_keys = ("y_pred", "y", "criterion_kwargs")
    _state_dict_all_req_keys = ("_sum", "_num_examples")

    def __init__(
        self,
        loss_fn: Callable,
        output_transform: Callable = lambda x: x,
        batch_size: Callable = len,
        device: Union[str, torch.device] = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        super(Loss, self).__init__(output_transform, device=device, skip_unrolling=skip_unrolling)
        self._loss_fn = loss_fn
        self._batch_size = batch_size

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[Union[torch.Tensor, Dict]]) -> None:
        if len(output) == 2:
            y_pred, y = cast(Tuple[torch.Tensor, torch.Tensor], output)
            kwargs: Dict = {}
        else:
            y_pred, y, kwargs = cast(Tuple[torch.Tensor, torch.Tensor, Dict], output)
        average_loss = self._loss_fn(y_pred, y, **kwargs).detach()

        if len(average_loss.shape) != 0:
            raise ValueError("loss_fn did not return the average loss.")

        n = self._batch_size(y)
        self._sum += average_loss.to(self._device) * n
        self._num_examples += n

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("Loss must have at least one example before it can be computed.")
        return self._sum.item() / self._num_examples
