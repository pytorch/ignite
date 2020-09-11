import itertools
from typing import Any, Callable, Union

import torch

from ignite.engine import Engine, Events
from ignite.metrics.metric import EpochWise, Metric, MetricUsage, reinit__is_reduced

__all__ = ["MetricsLambda"]


class MetricsLambda(Metric):
    """
    Apply a function to other metrics to obtain a new metric.
    The result of the new metric is defined to be the result
    of applying the function to the result of argument metrics.

    When update, this metric does not recursively update the metrics
    it depends on. When reset, all its dependency metrics would be
    resetted. When attach, all its dependency metrics would be attached
    automatically (but partially, e.g :meth:`~ignite.metrics.Metric.is_attached()` will return False).

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

    When check if the metric is attached, if one of its dependency
    metrics is detached, the metric is considered detached too.

    .. code-block:: python

        engine = ...
        precision = Precision(average=False)

        aP = precision.mean()

        aP.attach(engine, "aP")

        assert aP.is_attached(engine)
        # partially attached
        assert not precision.is_attached(engine)

        precision.detach(engine)

        assert not aP.is_attached(engine)
        # fully attached
        assert not precision.is_attached(engine)

    """

    def __init__(self, f: Callable, *args, **kwargs):
        self.function = f
        self.args = args
        self.kwargs = kwargs
        self.engine = None
        super(MetricsLambda, self).__init__(device="cpu")

    @reinit__is_reduced
    def reset(self) -> None:
        for i in itertools.chain(self.args, self.kwargs.values()):
            if isinstance(i, Metric):
                i.reset()

    @reinit__is_reduced
    def update(self, output) -> None:
        # NB: this method does not recursively update dependency metrics,
        # which might cause duplicate update issue. To update this metric,
        # users should manually update its dependencies.
        pass

    def compute(self) -> Any:
        materialized = [_get_value_on_cpu(i) for i in self.args]
        materialized_kwargs = {k: _get_value_on_cpu(v) for k, v in self.kwargs.items()}
        return self.function(*materialized, **materialized_kwargs)

    def _internal_attach(self, engine: Engine, usage: MetricUsage) -> None:
        self.engine = engine
        for index, metric in enumerate(itertools.chain(self.args, self.kwargs.values())):
            if isinstance(metric, MetricsLambda):
                metric._internal_attach(engine, usage)
            elif isinstance(metric, Metric):
                # NB : metrics is attached partially
                # We must not use is_attached() but rather if these events exist
                if not engine.has_event_handler(metric.started, usage.STARTED):
                    engine.add_event_handler(usage.STARTED, metric.started)
                if not engine.has_event_handler(metric.iteration_completed, usage.ITERATION_COMPLETED):
                    engine.add_event_handler(usage.ITERATION_COMPLETED, metric.iteration_completed)

    def attach(self, engine: Engine, name: str, usage: Union[str, MetricUsage] = EpochWise()) -> None:
        usage = self._check_usage(usage)
        # recursively attach all its dependencies (partially)
        self._internal_attach(engine, usage)
        # attach only handler on EPOCH_COMPLETED
        engine.add_event_handler(usage.COMPLETED, self.completed, name)

    def detach(self, engine: Engine, usage: Union[str, MetricUsage] = EpochWise()) -> None:
        usage = self._check_usage(usage)
        # remove from engine
        super(MetricsLambda, self).detach(engine, usage)
        self.engine = None

    def is_attached(self, engine: Engine, usage: Union[str, MetricUsage] = EpochWise()) -> bool:
        usage = self._check_usage(usage)
        # check recursively the dependencies
        return super(MetricsLambda, self).is_attached(engine, usage) and self._internal_is_attached(engine, usage)

    def _internal_is_attached(self, engine: Engine, usage: MetricUsage) -> bool:
        # if no engine, metrics is not attached
        if engine is None:
            return False
        # check recursively if metrics are attached
        is_detached = False
        for metric in itertools.chain(self.args, self.kwargs.values()):
            if isinstance(metric, MetricsLambda):
                if not metric._internal_is_attached(engine, usage):
                    is_detached = True
            elif isinstance(metric, Metric):
                if not engine.has_event_handler(metric.started, usage.STARTED):
                    is_detached = True
                if not engine.has_event_handler(metric.iteration_completed, usage.ITERATION_COMPLETED):
                    is_detached = True
        return not is_detached


def _get_value_on_cpu(v: Any):
    if isinstance(v, Metric):
        v = v.compute()
    if isinstance(v, torch.Tensor):
        v = v.cpu()
    return v
