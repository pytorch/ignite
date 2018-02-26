from abc import ABCMeta, abstractmethod

from ignite.engine import Events


class Metric(metaclass=ABCMeta):
    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, output):
        pass

    @abstractmethod
    def compute(self):
        pass

    def started(self, state):
        self.reset()

    def iteration_completed(self, state):
        self.update(state.output)

    def completed(self, state):
        state.metrics.append(self.compute())

    def attach(self, engine):
        engine.add_event_handler(Events.STARTED, self.started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(Events.COMPLETED, self.completed)
