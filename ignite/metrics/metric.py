from abc import ABCMeta, abstractmethod

from ignite.engines import Events


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

        This is called at the start of each evaluation run.
        """
        pass

    @abstractmethod
    def update(self, output):
        """
        Updates the metric's state using the passed batch output.

        This is called once for each batch of each evaluation run.

        Args:
            output: the is the output from the evaluator's process function
        """
        pass

    @abstractmethod
    def compute(self):
        """
        Computes the metric based on it's accumulated state.

        This is called at the end of each evaluation run.

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
        engine.add_event_handler(Events.STARTED, self.started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(Events.COMPLETED, self.completed, name)
