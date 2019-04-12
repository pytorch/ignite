from __future__ import division

import torch
import torch.distributed as dist
from ignite.metrics.metric import Metric


class DistributedMetric(Metric):
    def __init__(self, metric, world_size, device='cpu'):
        self._metric = metric
        self._world_size = world_size
        self._device = device
        self._val = None
        super().__init__()

    def update(self, output):
        self._metric.update(output)

    def reset(self):
        self._metric.reset()

    def compute(self):
        val = torch.tensor(self._metric.compute()).to(self._device)
        dist.all_reduce(val, op=dist.ReduceOp.SUM)

        return val / self._world_size
