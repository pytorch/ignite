from __future__ import division

from torch.autograd import Variable
from torch.nn.functional import nll_loss

from .metric import Metric
from ignite.exceptions import NotComputableError


class NegativeLogLikelihood(Metric):
    def __init__(self):
        self._sum = 0
        self._num_examples = 0

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        self._sum += nll_loss(Variable(y_pred), Variable(y)).data[0]
        self._num_examples += 1

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'NegativeLogLikelihood must have at least one example before it can be computed')
        return self._sum / self._num_examples
