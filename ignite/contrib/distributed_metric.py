from __future__ import division

import torch
import torch.distributed as dist
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


class DistributedLoss(Metric):
    """
    """

    def __init__(self, metric):
        super().__init__()
        self._metric = metric
        self._sum = torch.zeros(1)
        self._num_examples = torch.zeros(1)

    def reset(self):
        self._metric.reset()

    def update(self, output):
        self._metric.update(output)

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed.')

        dist.all_reduce(self._sum, op=dist.reduce_op.SUM)
        dist.all_reduce(self._num_examples, op=dist.reduce_op.SUM)
        return (self._sum / self._num_examples).item()

