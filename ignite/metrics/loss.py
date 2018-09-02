from __future__ import division

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


class Loss(Metric):
    """
    Calculates the average loss according to the passed loss_fn.

    - `loss_fn` must return the average loss over all observations in the batch.
    - `update` must receive output of the form `(y_pred, y)`.
    """
    def __init__(self, loss_fn, output_transform=lambda x: x):
        super(Loss, self).__init__(output_transform)
        self._loss_fn = loss_fn

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        average_loss = self._loss_fn(y_pred, y)
        assert len(average_loss.shape) == 0, '`loss_fn` did not return the average loss'
        self._sum += average_loss.item() * y.shape[0]
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed')
        return self._sum / self._num_examples
