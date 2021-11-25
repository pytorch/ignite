from typing import Callable, Dict, Sequence, Tuple, Union, cast

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

        .. testcode::

            model = nn.Linear(10, 3)

            criterion = nll_loss
            c_kwargs = {"reduction": "sum"}

            def process_function(engine, batch):
                y_pred, y = batch
                return y_pred, y

            def output_transform(output):
                y_pred, y = output
                criterion_kwargs = c_kwargs
                return y_pred, y, c_kwargs 

            engine = Engine(process_function)
            metric = Loss(criterion, output_transform=output_transform)
            metric.attach(engine, 'loss')
            y_pred = torch.Tensor([
                    [0.6700, 0.9327, 0.3307, 0.9657, 0.5026, 0.5136, 0.4327, 0.3655, 0.5784, 0.7256],
                    [0.7358, 0.4609, 0.5122, 0.1045, 0.0030, 0.4085, 0.7060, 0.7697, 0.1859, 0.2599],
                    [0.1652, 0.6846, 0.1873, 0.6037, 0.4760, 0.8666, 0.2732, 0.7021, 0.4156, 0.6474],
                    [0.5057, 0.2380, 0.4412, 0.0578, 0.3471, 0.7685, 0.7794, 0.7815, 0.4093, 0.2081]])
            y_true = torch.LongTensor([2, 1, 2, 2])
            state = engine.run([[y_pred, y_true]])
            print(state.metrics['loss'])

        .. testoutput::

            -1.4201000...

    """

    required_output_keys = ("y_pred", "y", "criterion_kwargs")

    print(required_output_keys)

    def __init__(
        self,
        loss_fn: Callable,
        output_transform: Callable = lambda x: x,
        batch_size: Callable = len,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(Loss, self).__init__(output_transform, device=device)
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
            kwargs = {}  # type: Dict
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
