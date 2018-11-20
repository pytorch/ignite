from abc import ABCMeta, abstractmethod
from six import with_metaclass
from ignite.engine import Events
from ignite.exceptions import NotComputableError
import functools
import torch


class MetricMeta(ABCMeta):
    """Metaclass for creating metric classes.

    This class does the following to metric classes:

    - Hijack code to prevent metric from being ``compute``d before ``update``d.
    """

    def __init__(cls, name, bases, dct):
        super(MetricMeta, cls).__init__(name, bases, dct)

        old_reset = cls.reset
        old_update = cls.update
        old_compute = cls.compute

        def wrapped_reset(self):
            self._updated = False
            old_reset(self)
        cls.reset = functools.wraps(cls.reset)(wrapped_reset)

        def wrapped_update(self, output):
            self._updated = True
            old_update(self, output)
        cls.update = functools.wraps(cls.update)(wrapped_update)

        def wrapped_compute(self):
            if not self._updated:
                raise NotComputableError('not updated before compute')
            return old_compute(self)
        cls.compute = functools.wraps(cls.compute)(wrapped_compute)


class Metric(with_metaclass(MetricMeta)):
    """
    Base class for all Metrics.

    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.

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

    @torch.no_grad()
    def iteration_completed(self, engine):
        output = self._output_transform(engine.state.output)
        self.update(output)

    def completed(self, engine, name):
        engine.state.metrics[name] = self.compute()

    def attach(self, engine, name):
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed, name)
        if not engine.has_event_handler(self.started, Events.EPOCH_STARTED):
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
