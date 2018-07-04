from abc import ABCMeta, abstractmethod

import torch

from ignite.engine import Events


class Metric(object):
    __metaclass__ = ABCMeta

    """
    Base class for all Metrics.

    Metrics provide a way to compute various quantities of interest in an online
    fashion without having to store the entire output history of a model.

    Args:
        output_transform (callable): a callable that is used to transform the
            model's output into the form expected by the metric. This can be
            useful if, for example, you have a multi-output model and you want to
            compute the metric with respect to one of the outputs.
    """
    def __init__(self, output_transform=lambda x: x):
        self._output_transform = output_transform
        self.reset()

    @abstractmethod
    def reset(self):
        """
        Resets the metric to to it's initial state.

        This is called at the start of each epoch.
        """
        pass

    @abstractmethod
    def update(self, output):
        """
        Updates the metric's state using the passed batch output.

        This is called once for each batch.

        Args:
            output: the is the output from the engine's process function
        """
        pass

    @abstractmethod
    def compute(self):
        """
        Computes the metric based on it's accumulated state.

        This is called at the end of each epoch.

        Returns:
            Any: the actual quantity of interest

        Raises:
            NotComputableError: raised when the metric cannot be computed
        """
        pass

    def started(self, engine):
        self.reset()

    def iteration_completed(self, engine):
        output = self._output_transform(engine.state.output)
        self.update(output)

    def completed(self, engine, name):
        engine.state.metrics[name] = self.compute()

    def attach(self, engine, name):
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed, name)


class EpochMetric(Metric):
    """Base class for metrics that should be computed on the entire output history of a model.
    Model's output and targets are restricted to be of shape `(batch_size, n_classes)`.

    If target shape is `(batch_size, n_classes)` and `n_classes > 1` than it should be binary: e.g. `[[0, 1, 0, 1], ]`
    """

    def reset(self):
        self._predictions = torch.tensor([], dtype=torch.float32)
        self._targets = torch.tensor([], dtype=torch.long)

    def update(self, output):
        y_pred, y = output

        assert 1 <= y_pred.ndimension() <= 2, "Predictions should be of shape (batch_size, n_classes)"
        assert 1 <= y.ndimension() <= 2, "Targets should be of shape (batch_size, n_classes)"

        if y.ndimension() == 2:
            assert torch.equal(y ** 2, y), 'Targets should be binary (0 or 1)'

        if y_pred.ndimension() == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(dim=-1)

        if y.ndimension() == 2 and y.shape[1] == 1:
            y = y.squeeze(dim=-1)

        y_pred = y_pred.type_as(self._predictions)
        y = y.type_as(self._targets)

        self._predictions = torch.cat([self._predictions, y_pred], dim=0)
        self._targets = torch.cat([self._targets, y], dim=0)

    @abstractmethod
    def compute(self):
        """
        Computes the metric based on it's accumulated state.

        This is called at the end of each epoch.

        Returns:
            Any: the actual quantity of interest

        Raises:
            NotComputableError: raised when the metric cannot be computed
        """
        pass
