import itertools
from typing import Callable, Any
import weakref

from ignite.metrics.metric import Metric, reinit__is_reduced
from ignite.engine import Events, Engine

__all__ = ["MetricsLambda"]


class MetricsLambda(Metric):
    """
    Apply a function to other metrics to obtain a new metric.
    The result of the new metric is defined to be the result
    of applying the function to the result of argument metrics.

    When update, this metric does not recursively update the metrics
    it depends on. When reset, all its dependency metrics would be
    resetted. When attach, all its dependencies would be automatically
    attached.

    Args:
        f (callable): the function that defines the computation
        args (sequence): Sequence of other metrics or something
            else that will be fed to ``f`` as arguments.

    Example:

    .. code-block:: python

        precision = Precision(average=False)
        recall = Recall(average=False)

        def Fbeta(r, p, beta):
            return torch.mean((1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-20)).item()

        F1 = MetricsLambda(Fbeta, recall, precision, 1)
        F2 = MetricsLambda(Fbeta, recall, precision, 2)
        F3 = MetricsLambda(Fbeta, recall, precision, 3)
        F4 = MetricsLambda(Fbeta, recall, precision, 4)
    """

    def __init__(self, f: Callable, *args, **kwargs):
        self.function = f
        self.args = args
        self.kwargs = kwargs
        self.engine = None
        super(MetricsLambda, self).__init__(device="cpu")

    @reinit__is_reduced
    def reset(self) -> None:
        if not self._detach_if_necessary(self.engine):
            for i in itertools.chain(self.args, self.kwargs.values()):
                if isinstance(i, Metric):
                    i.reset()

    @reinit__is_reduced
    def update(self, output) -> None:
        # NB: this method does not recursively update dependency metrics,
        # which might cause duplicate update issue. To update this metric,
        # users should manually update its dependencies.
        self._detach_if_necessary(self.engine)

    def compute(self) -> Any:
        if not self._detach_if_necessary(self.engine):
            materialized = [i.compute() if isinstance(i, Metric) else i for i in self.args]
            materialized_kwargs = {k: (v.compute() if isinstance(v, Metric) else v) for k, v in self.kwargs.items()}
            return self.function(*materialized, **materialized_kwargs)

    def _internal_attach(self, engine: Engine) -> None:
        self.engine = weakref.ref(engine)
        for index, metric in enumerate(itertools.chain(self.args, self.kwargs.values())):
            if isinstance(metric, MetricsLambda):
                metric._internal_attach(engine)
            elif isinstance(metric, Metric):
                # NB : metrics is attached partially
                # We must not use is_attached() but rather if these events exist
                if not engine.has_event_handler(metric.started, Events.EPOCH_STARTED):
                    engine.add_event_handler(Events.EPOCH_STARTED, metric.started)
                if not engine.has_event_handler(metric.iteration_completed, Events.ITERATION_COMPLETED):
                    engine.add_event_handler(Events.ITERATION_COMPLETED, metric.iteration_completed)

    def attach(self, engine: Engine, name: str) -> None:
        # recursively attach all its dependencies (partially)
        self._internal_attach(engine)
        # attach only handler on EPOCH_COMPLETED
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed, name)

    def is_attached(self, engine: Engine) -> bool:
        self._detach_if_necessary(weakref.ref(engine))
        return super().is_attached(engine)

    def _detach_if_necessary(self, engine: weakref) -> bool:
        if engine is None or engine() is None:
            return False
        need_detach = False
        for metric in itertools.chain(self.args, self.kwargs.values()):
            if isinstance(metric, Metric):
                if not engine().has_event_handler(metric.started, Events.EPOCH_STARTED):
                    need_detach = True
                if not engine().has_event_handler(metric.iteration_completed, Events.ITERATION_COMPLETED):
                    need_detach = True
        if need_detach:
            super().detach(engine())
        return need_detach
