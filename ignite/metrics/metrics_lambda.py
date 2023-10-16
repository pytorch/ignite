import itertools
from typing import Any, Callable, Optional, Union

import torch

from ignite.engine import Engine
from ignite.metrics.metric import EpochWise, Metric, MetricUsage, reinit__is_reduced

__all__ = ["MetricsLambda"]


class MetricsLambda(Metric):
    """
    Apply a function to other metrics to obtain a new metric.
    The result of the new metric is defined to be the result
    of applying the function to the result of argument metrics.

    When update, this metric recursively updates the metrics
    it depends on. When reset, all its dependency metrics would be
    resetted as well. When attach, all its dependency metrics would be attached
    automatically (but partially, e.g :meth:`~ignite.metrics.metric.Metric.is_attached()` will return False).

    Args:
        f: the function that defines the computation
        args: Sequence of other metrics or something
            else that will be fed to ``f`` as arguments.
        kwargs: Sequence of other metrics or something
            else that will be fed to ``f`` as keyword arguments.

    Examples:

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

        .. include:: defaults.rst
            :start-after: :orphan:

        .. testcode::

            precision = Precision(average=False)
            recall = Recall(average=False)

            def Fbeta(r, p, beta):
                return torch.mean((1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-20)).item()

            F1 = MetricsLambda(Fbeta, recall, precision, 1)
            F2 = MetricsLambda(Fbeta, recall, precision, 2)
            F3 = MetricsLambda(Fbeta, recall, precision, 3)
            F4 = MetricsLambda(Fbeta, recall, precision, 4)

            F1.attach(default_evaluator, "F1")
            F2.attach(default_evaluator, "F2")
            F3.attach(default_evaluator, "F3")
            F4.attach(default_evaluator, "F4")

            y_true = torch.tensor([1, 0, 1, 0, 0, 1])
            y_pred = torch.tensor([1, 0, 1, 0, 1, 1])
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["F1"])
            print(state.metrics["F2"])
            print(state.metrics["F3"])
            print(state.metrics["F4"])

        .. testoutput::

            0.8571...
            0.9375...
            0.9677...
            0.9807...

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

    _state_dict_all_req_keys = ("_updated", "args", "kwargs")

    def __init__(self, f: Callable, *args: Any, **kwargs: Any) -> None:
        self.function = f
        self.args = list(args)  # we need args to be a list instead of a tuple for state_dict/load_state_dict feature
        self.kwargs = kwargs
        self.engine: Optional[Engine] = None
        self._updated = False
        super(MetricsLambda, self).__init__(device="cpu")

    @reinit__is_reduced
    def reset(self) -> None:
        for i in itertools.chain(self.args, self.kwargs.values()):
            if isinstance(i, Metric):
                i.reset()
        self._updated = False

    @reinit__is_reduced
    def update(self, output: Any) -> None:
        if self.engine:
            raise ValueError(
                "MetricsLambda is already attached to an engine, "
                "and MetricsLambda can't use update API while it's attached."
            )

        for i in itertools.chain(self.args, self.kwargs.values()):
            if isinstance(i, Metric):
                i.update(output)

        self._updated = True

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
        if self._updated:
            raise ValueError(
                "The underlying metrics are already updated, can't attach while using reset/update/compute API."
            )
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


def _get_value_on_cpu(v: Any) -> Any:
    if isinstance(v, Metric):
        v = v.compute()
    if isinstance(v, torch.Tensor):
        v = v.cpu()
    return v
