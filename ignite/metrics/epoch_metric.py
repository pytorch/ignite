import warnings
from typing import Callable, Sequence

import torch

from ignite.metrics.metric import Metric

__all__ = ["EpochMetric"]


class EpochMetric(Metric):
    """Class for metrics that should be computed on the entire output history of a model.
    Model's output and targets are restricted to be of shape `(batch_size, n_classes)`. Output
    datatype should be `float32`. Target datatype should be `long`.

    .. warning::

        Current implementation stores all input data (output and target) in as tensors before computing a metric.
        This can potentially lead to a memory error if the input data is larger than available RAM.

    .. warning::

        Current implementation does not work with distributed computations. Results are not gather across all devices
        and computed results are valid for a single device only.

    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.

    If target shape is `(batch_size, n_classes)` and `n_classes > 1` than it should be binary: e.g. `[[0, 1, 0, 1], ]`.

    Args:
        compute_fn (callable): a callable with the signature (`torch.tensor`, `torch.tensor`) takes as the input
            `predictions` and `targets` and returns a scalar.
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.

    """

    def __init__(self, compute_fn: Callable, output_transform: Callable = lambda x: x):

        if not callable(compute_fn):
            raise TypeError("Argument compute_fn should be callable.")

        super(EpochMetric, self).__init__(output_transform=output_transform, device="cpu")
        self.compute_fn = compute_fn

    def reset(self) -> None:
        self._predictions = []
        self._targets = []

    def _check_shape(self, output):
        y_pred, y = output
        if y_pred.ndimension() not in (1, 2):
            raise ValueError("Predictions should be of shape (batch_size, n_classes) or (batch_size, ).")

        if y.ndimension() not in (1, 2):
            raise ValueError("Targets should be of shape (batch_size, n_classes) or (batch_size, ).")

        if y.ndimension() == 2:
            if not torch.equal(y ** 2, y):
                raise ValueError("Targets should be binary (0 or 1).")

    def _check_type(self, output):
        y_pred, y = output
        if len(self._predictions) < 1:
            return
        dtype_preds = self._predictions[-1].type()
        if dtype_preds != y_pred.type():
            raise ValueError(
                "Incoherent types between input y_pred and stored predictions: "
                "{} vs {}".format(dtype_preds, y_pred.type())
            )

        dtype_targets = self._targets[-1].type()
        if dtype_targets != y.type():
            raise ValueError(
                "Incoherent types between input y and stored targets: " "{} vs {}".format(dtype_targets, y.type())
            )

    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        y_pred, y = output
        if y_pred.ndimension() == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=-1)

        if y.ndimension() == 2 and y.shape[1] == 1:
            y = y.squeeze(dim=-1)

        y_pred = y_pred.cpu().clone()
        y = y.cpu().clone()

        self._check_type((y_pred, y))
        self._predictions.append(y_pred)
        self._targets.append(y)

        # Check once the signature and execution of compute_fn
        if len(self._predictions) == 1:
            try:
                self.compute_fn(self._predictions[0], self._targets[0])
            except Exception as e:
                warnings.warn("Probably, there can be a problem with `compute_fn`:\n {}.".format(e), EpochMetricWarning)

    def compute(self) -> None:
        _prediction_tensor = torch.cat(self._predictions, dim=0)
        _target_tensor = torch.cat(self._targets, dim=0)
        return self.compute_fn(_prediction_tensor, _target_tensor)


class EpochMetricWarning(UserWarning):
    pass
