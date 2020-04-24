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

    def _update(self, output):
        EpochMetric.update(self, output)
