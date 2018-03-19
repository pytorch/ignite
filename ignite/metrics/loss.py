from __future__ import division

from torch.autograd import Variable

from .metric import Metric
from ignite.exceptions import NotComputableError


class Loss(Metric):
    """
    Calculates the average loss according to the passed loss_fn.

    `update` must receive output of the form (y_pred, y).
    """
    def __init__(self, loss_fn):
        super(Loss, self).__init__()
        self._loss_fn = loss_fn

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        average_loss = self._loss_fn(Variable(y_pred, volatile=True), Variable(y, volatile=True))
        self._sum += average_loss.data[0] * y.shape[0]
        self._num_examples += y.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed')
        return self._sum / self._num_examples
