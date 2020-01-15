from __future__ import division

from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class Loss(Metric):
    """
    Calculates the average loss according to the passed loss_fn.

    Args:
        loss_fn (callable): a callable taking a prediction tensor, a target
            tensor, optionally other arguments, and returns the average loss
            over all observations in the batch.
        output_transform (callable): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric.
            This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            The output is expected to be a tuple `(prediction, target)` or
            (prediction, target, kwargs) where kwargs is a dictionary of extra
            keywords arguments. If extra keywords arguments are provided they are passed to `loss_fn`.
        batch_size (callable): a callable taking a target tensor that returns the
            first dimension size (usually the batch size).
        device (str of torch.device, optional): device specification in case of distributed computation usage.
            In most of the cases, it can be defined as "cuda:local_rank" or "cuda"
            if already set `torch.cuda.set_device(local_rank)`. By default, if a distributed process group is
            initialized and available, device is set to `cuda`.

    """
    _required_output_keys = None

    def __init__(self, loss_fn, output_transform=lambda x: x,
                 batch_size=lambda x: len(x), device=None):
        super(Loss, self).__init__(output_transform, device=device)
        self._loss_fn = loss_fn
        self._batch_size = batch_size

    @reinit__is_reduced
    def reset(self):
        self._sum = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        if len(output) == 2:
            y_pred, y = output
            kwargs = {}
        else:
            y_pred, y, kwargs = output
        average_loss = self._loss_fn(y_pred, y, **kwargs)

        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')

        N = self._batch_size(y)
        self._sum += average_loss.item() * N
        self._num_examples += N

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed.')
        return self._sum / self._num_examples
