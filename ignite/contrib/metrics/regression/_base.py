import warnings
import torch
from abc import abstractmethod
from ignite.metrics import Metric


class _BaseRegression(Metric):
    # Base class for all regression metrics
    # `update` method check the shapes and call internal overloaded
    # method `_update`

    def update(self, output):
        y_pred, y = output
        if y_pred.shape != y.shape:
            raise ValueError("Input data shapes should be the same, but given {} and {}"
                             .format(y_pred.shape, y.shape))

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


class _BaseRegressionEpoch(Metric):
    """Class for regression metrics that should be computed on the entire output history of a model.
    Model's output and targets are restricted to be of shape `(batch_size, 1) or (batch_size, )`. Output
    datatype should be `float32`. Target datatype should be `long`.

    .. warning::

        Current implementation stores all input data (output and target) in as tensors before computing a metric.
        This can potentially lead to a memory error if the input data is larger than available RAM.


    - `update` must receive output of the form `(y_pred, y)`.

    Args:
        compute_fn (callable): a callable with the signature (`torch.tensor`, `torch.tensor`) takes as the input
            `predictions` and `targets` and returns a scalar.
        output_transform (callable, optional): a callable that is used to transform the
            :class:`ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.

    """

    def __init__(self, compute_fn, output_transform=lambda x: x):

        if not callable(compute_fn):
            raise TypeError("Argument compute_fn should be callable")

        super(_BaseRegressionEpoch, self).__init__(output_transform=output_transform)
        self.compute_fn = compute_fn

    def reset(self):
        self._predictions = torch.tensor([], dtype=torch.float32)
        self._targets = torch.tensor([], dtype=torch.float32)

    def update(self, output):
        y_pred, y = output
        if y_pred.shape != y.shape:
            raise ValueError("Input data shapes should be the same, but given {} and {}"
                             .format(y_pred.shape, y.shape))

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

        y_pred = y_pred.type_as(self._predictions)
        y = y.type_as(self._targets)

        self._predictions = torch.cat([self._predictions, y_pred], dim=0)
        self._targets = torch.cat([self._targets, y], dim=0)

        # Check once the signature and execution of compute_fn
        if self._predictions.shape == y_pred.shape:
            try:
                self.compute_fn(self._predictions, self._targets)
            except Exception as e:
                warnings.warn("Probably, there can be a problem with `compute_fn`:\n {}".format(e),
                              RuntimeWarning)

    def compute(self):
        return self.compute_fn(self._predictions, self._targets)
