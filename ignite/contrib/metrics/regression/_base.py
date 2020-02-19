from abc import abstractmethod

import warnings

import torch

from ignite.metrics import Metric, EpochMetric


class _BaseRegression(Metric):
    # Base class for all regression metrics
    # `update` method check the shapes and call internal overloaded
    # method `_update`.

    def update(self, output):
        y_pred, y = output
        if y_pred.shape != y.shape:
            raise ValueError("Input data shapes should be the same, but given {} and {}".format(y_pred.shape, y.shape))

        c1 = y_pred.ndimension() == 2 and y_pred.shape[1] == 1
        if not (y_pred.ndimension() == 1 or c1):
            raise ValueError("Input y_pred should have shape (N,) or (N, 1), but given {}".format(y_pred.shape))

        c2 = y.ndimension() == 2 and y.shape[1] == 1
        if not (y.ndimension() == 1 or c2):
            raise ValueError("Input y should have shape (N,) or (N, 1), but given {}".format(y.shape))

        if c1:
            y_pred = y_pred.squeeze(dim=-1)

        if c2:
            y = y.squeeze(dim=-1)

        self._update((y_pred, y))

    @abstractmethod
    def _update(self, output):
        pass


class _BaseRegressionEpoch(_BaseRegression, EpochMetric):
    # Base class for all median-based regression metrics
    # `update` method check the shapes and call internal overloaded method `_update`.
    # Class internally stores complete history of predictions and targets of type float32.

    def __init__(self, compute_fn, output_transform=lambda x: x):
        EpochMetric.__init__(self, compute_fn=compute_fn, output_transform=output_transform)

    def reset(self):
        self._predictions = torch.tensor([], dtype=torch.float32)
        self._targets = torch.tensor([], dtype=torch.float32)

    def _update(self, output):
        y_pred, y = output
        y_pred = y_pred.type_as(self._predictions)
        y = y.type_as(self._targets)

        self._predictions = torch.cat([self._predictions, y_pred], dim=0)
        self._targets = torch.cat([self._targets, y], dim=0)

        # Check once the signature and execution of compute_fn
        if self._predictions.shape == y_pred.shape:
            try:
                self.compute_fn(self._predictions, self._targets)
            except Exception as e:
                warnings.warn("Probably, there can be a problem with `compute_fn`:\n {}".format(e), RuntimeWarning)
