import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import torch

import ignite.distributed as idist
from ignite.engine import CallableEventWithFilter, Engine, Events

if TYPE_CHECKING:
    from ignite.metrics.metrics_lambda import MetricsLambda

__all__ = ["Metric", "MetricUsage", "EpochWise", "BatchWise", "BatchFiltered"]


class MetricUsage:
    """
    Base class for all usages of metrics.

    A usage of metric defines the events when a metric starts to compute, updates and completes.
    Valid events are from :class:`~ignite.engine.events.Events`.

    Args:
        started: event when the metric starts to compute. This event will be associated to
            :meth:`~ignite.metrics.Metric.started`.
        completed: event when the metric completes. This event will be associated to
            :meth:`~ignite.metrics.Metric.completed`.
        iteration_completed: event when the metric updates. This event will be associated to
            :meth:`~ignite.metrics.Metric.iteration_completed`.
    """

    def __init__(self, started: Events, completed: Events, iteration_completed: CallableEventWithFilter) -> None:
        self.__started = started
        self.__completed = completed
        self.__iteration_completed = iteration_completed

    @property
    def STARTED(self) -> Events:
        return self.__started

    @property
    def COMPLETED(self) -> Events:
        return self.__completed

    @property
    def ITERATION_COMPLETED(self) -> CallableEventWithFilter:
        return self.__iteration_completed


class EpochWise(MetricUsage):
    """
    Epoch-wise usage of Metrics. It's the default and most common usage of metrics.

    Metric's methods are triggered on the following engine events:

    - :meth:`~ignite.metrics.Metric.started` on every ``EPOCH_STARTED`` (See :class:`~ignite.engine.events.Events`).
    - :meth:`~ignite.metrics.Metric.iteration_completed` on every ``ITERATION_COMPLETED``.
    - :meth:`~ignite.metrics.Metric.completed` on every ``EPOCH_COMPLETED``.

    Attributes:
        usage_name (str): usage name string
    """

    usage_name = "epoch_wise"

    def __init__(self) -> None:
        super(EpochWise, self).__init__(
            started=Events.EPOCH_STARTED,
            completed=Events.EPOCH_COMPLETED,
            iteration_completed=Events.ITERATION_COMPLETED,
        )


class BatchWise(MetricUsage):
    """
    Batch-wise usage of Metrics.

    Metric's methods are triggered on the following engine events:

    - :meth:`~ignite.metrics.Metric.started` on every ``ITERATION_STARTED`` (See :class:`~ignite.engine.events.Events`).
    - :meth:`~ignite.metrics.Metric.iteration_completed` on every ``ITERATION_COMPLETED``.
    - :meth:`~ignite.metrics.Metric.completed` on every ``ITERATION_COMPLETED``.

    Attributes:
        usage_name (str): usage name string
    """

    usage_name = "batch_wise"

    def __init__(self) -> None:
        super(BatchWise, self).__init__(
            started=Events.ITERATION_STARTED,
            completed=Events.ITERATION_COMPLETED,
            iteration_completed=Events.ITERATION_COMPLETED,
        )


class BatchFiltered(MetricUsage):
    """
    Batch filtered usage of Metrics. This usage is similar to epoch-wise but update event is filtered.

    Metric's methods are triggered on the following engine events:

    - :meth:`~ignite.metrics.Metric.started` on every ``EPOCH_STARTED`` (See :class:`~ignite.engine.events.Events`).
    - :meth:`~ignite.metrics.Metric.iteration_completed` on filtered ``ITERATION_COMPLETED``.
    - :meth:`~ignite.metrics.Metric.completed` on every ``EPOCH_COMPLETED``.

    Args:
        *args: Positional arguments to setup :attr:`~ignite.engine.events.Events.ITERATION_COMPLETED(*args, **kwargs)`
        **kwargs: Keyword arguments to setup :attr:`~ignite.engine.events.Events.ITERATION_COMPLETED(*args, **kwargs)`
            handled by :meth:`~ignite.metrics.Metric.iteration_completed`.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(BatchFiltered, self).__init__(
            started=Events.EPOCH_STARTED,
            completed=Events.EPOCH_COMPLETED,
            iteration_completed=Events.ITERATION_COMPLETED(*args, **kwargs),
        )


class Metric(metaclass=ABCMeta):
    """
    Base class for all Metrics.

    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device (str or torch.device): specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.

    Attributes:
        required_output_keys (tuple): dictionary defines required keys to be found in ``engine.state.output`` if the
            latter is a dictionary. Default, ``("y_pred", "y")``. This is useful with custom metrics that can require
            other arguments than predictions ``y_pred`` and targets ``y``. See notes below for an example.

    Note:

        Let's implement a custom metric that requires ``y_pred``, ``y`` and ``x`` as input for ``update`` function.
        In the example below we show how to setup standard metric like Accuracy and the custom metric using by an
        ``evaluator`` created with :meth:`~ignite.engine.create_supervised_evaluator` method.

        .. code-block:: python

            # https://discuss.pytorch.org/t/how-access-inputs-in-custom-ignite-metric/91221/5

            import torch
            import torch.nn as nn

            from ignite.metrics import Metric, Accuracy
            from ignite.engine import create_supervised_evaluator

            class CustomMetric(Metric):

                required_output_keys = ("y_pred", "y", "x")

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)

                def update(self, output):
                    y_pred, y, x = output
                    # ...

                def reset(self):
                    # ...
                    pass

                def compute(self):
                    # ...
                    pass

            model = ...

            metrics = {
                "Accuracy": Accuracy(),
                "CustomMetric": CustomMetric()
            }

            evaluator = create_supervised_evaluator(
                model,
                metrics=metrics,
                output_transform=lambda x, y, y_pred: {"x": x, "y": y, "y_pred": y_pred}
            )

            res = evaluator.run(data)

    """

    # public class attribute
    required_output_keys = ("y_pred", "y")  # type: Optional[Tuple]
    # for backward compatibility
    _required_output_keys = required_output_keys

    def __init__(
        self, output_transform: Callable = lambda x: x, device: Union[str, torch.device] = torch.device("cpu"),
    ):
        self._output_transform = output_transform

        # Check device if distributed is initialized:
        if idist.get_world_size() > 1:

            # check if reset and update methods are decorated. Compute may not be decorated
            if not (hasattr(self.reset, "_decorated") and hasattr(self.update, "_decorated")):
                warnings.warn(
                    "{} class does not support distributed setting. Computed result is not collected "
                    "across all computing devices".format(self.__class__.__name__),
                    RuntimeWarning,
                )

        # Some metrics have a large performance regression when run on XLA devices, so for now, we disallow it.
        if torch.device(device).type == "xla":
            raise ValueError("Cannot create metric on an XLA device. Use device='cpu' instead.")

        self._device = torch.device(device)
        self._is_reduced = False
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the metric to it's initial state.

        By default, this is called at the start of each epoch.
        """
        pass

    @abstractmethod
    def update(self, output: Any) -> None:
        """
        Updates the metric's state using the passed batch output.

        By default, this is called once for each batch.

        Args:
            output: the is the output from the engine's process function.
        """
        pass

    @abstractmethod
    def compute(self) -> Any:
        """
        Computes the metric based on it's accumulated state.

        By default, this is called at the end of each epoch.

        Returns:
            Any: | the actual quantity of interest. However, if a :class:`~collections.abc.Mapping` is returned,
                 it will be (shallow) flattened into `engine.state.metrics` when
                 :func:`~ignite.metrics.Metric.completed` is called.

        Raises:
            NotComputableError: raised when the metric cannot be computed.
        """
        pass

    def started(self, engine: Engine) -> None:
        """Helper method to start data gathering for metric's computation. It is automatically attached to the
        `engine` with :meth:`~ignite.metrics.Metric.attach`.

        Args:
            engine (Engine): the engine to which the metric must be attached
        """
        self.reset()

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        """Helper method to update metric's computation. It is automatically attached to the
        `engine` with :meth:`~ignite.metrics.Metric.attach`.

        Args:
            engine (Engine): the engine to which the metric must be attached
        """

        output = self._output_transform(engine.state.output)
        if isinstance(output, Mapping):
            if self.required_output_keys is None:
                raise TypeError(
                    "Transformed engine output for {} metric should be a tuple/list, but given {}".format(
                        self.__class__.__name__, type(output)
                    )
                )
            if not all([k in output for k in self.required_output_keys]):
                raise ValueError(
                    "When transformed engine's output is a mapping, "
                    "it should contain {} keys, but given {}".format(self.required_output_keys, list(output.keys()))
                )
            output = tuple(output[k] for k in self.required_output_keys)
        self.update(output)

    def completed(self, engine: Engine, name: str) -> None:
        """Helper method to compute metric's value and put into the engine. It is automatically attached to the
        `engine` with :meth:`~ignite.metrics.Metric.attach`.

        Args:
            engine (Engine): the engine to which the metric must be attached
        """
        result = self.compute()
        if isinstance(result, Mapping):
            for key, value in result.items():
                engine.state.metrics[key] = value
        else:
            if isinstance(result, torch.Tensor) and len(result.size()) == 0:
                result = result.item()

            engine.state.metrics[name] = result

    def _check_usage(self, usage: Union[str, MetricUsage]) -> MetricUsage:
        if isinstance(usage, str):
            if usage == EpochWise.usage_name:
                usage = EpochWise()
            elif usage == BatchWise.usage_name:
                usage = BatchWise()
            else:
                raise ValueError(
                    "usage should be 'EpochWise.usage_name' or 'BatchWise.usage_name', get {}".format(usage)
                )
        if not isinstance(usage, MetricUsage):
            raise TypeError("Unhandled usage type {}".format(type(usage)))
        return usage

    def attach(self, engine: Engine, name: str, usage: Union[str, MetricUsage] = EpochWise()) -> None:
        """
        Attaches current metric to provided engine. On the end of engine's run, `engine.state.metrics` dictionary will
        contain computed metric's value under provided name.

        Args:
            engine (Engine): the engine to which the metric must be attached
            name (str): the name of the metric to attach
            usage (str or MetricUsage, optional): the usage of the metric. Valid string values should be
                :attr:`ignite.metrics.EpochWise.usage_name` (default) or
                :attr:`ignite.metrics.BatchWise.usage_name`.

        Example:

        .. code-block:: python

            metric = ...
            metric.attach(engine, "mymetric")

            assert "mymetric" in engine.run(data).metrics

            assert metric.is_attached(engine)

        Example with usage:

        .. code-block:: python

            metric = ...
            metric.attach(engine, "mymetric", usage=BatchWise.usage_name)

            assert "mymetric" in engine.run(data).metrics

            assert metric.is_attached(engine, usage=BatchWise.usage_name)
        """
        usage = self._check_usage(usage)
        if not engine.has_event_handler(self.started, usage.STARTED):
            engine.add_event_handler(usage.STARTED, self.started)
        if not engine.has_event_handler(self.iteration_completed, usage.ITERATION_COMPLETED):
            engine.add_event_handler(usage.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(usage.COMPLETED, self.completed, name)

    def detach(self, engine: Engine, usage: Union[str, MetricUsage] = EpochWise()) -> None:
        """
        Detaches current metric from the engine and no metric's computation is done during the run.
        This method in conjunction with :meth:`~ignite.metrics.Metric.attach` can be useful if several
        metrics need to be computed with different periods. For example, one metric is computed every training epoch
        and another metric (e.g. more expensive one) is done every n-th training epoch.

        Args:
            engine (Engine): the engine from which the metric must be detached
            usage (str or MetricUsage, optional): the usage of the metric. Valid string values should be
                'epoch_wise' (default) or 'batch_wise'.

        Example:

        .. code-block:: python

            metric = ...
            engine = ...
            metric.detach(engine)

            assert "mymetric" not in engine.run(data).metrics

            assert not metric.is_attached(engine)

        Example with usage:

        .. code-block:: python

            metric = ...
            engine = ...
            metric.detach(engine, usage="batch_wise")

            assert "mymetric" not in engine.run(data).metrics

            assert not metric.is_attached(engine, usage="batch_wise")
        """
        usage = self._check_usage(usage)
        if engine.has_event_handler(self.completed, usage.COMPLETED):
            engine.remove_event_handler(self.completed, usage.COMPLETED)
        if engine.has_event_handler(self.started, usage.STARTED):
            engine.remove_event_handler(self.started, usage.STARTED)
        if engine.has_event_handler(self.iteration_completed, usage.ITERATION_COMPLETED):
            engine.remove_event_handler(self.iteration_completed, usage.ITERATION_COMPLETED)

    def is_attached(self, engine: Engine, usage: Union[str, MetricUsage] = EpochWise()) -> bool:
        """
        Checks if current metric is attached to provided engine. If attached, metric's computed
        value is written to `engine.state.metrics` dictionary.

        Args:
            engine (Engine): the engine checked from which the metric should be attached
            usage (str or MetricUsage, optional): the usage of the metric. Valid string values should be
                'epoch_wise' (default) or 'batch_wise'.
        """
        usage = self._check_usage(usage)
        return engine.has_event_handler(self.completed, usage.COMPLETED)

    def __add__(self, other: Any) -> "MetricsLambda":
        from ignite.metrics.metrics_lambda import MetricsLambda

        return MetricsLambda(lambda x, y: x + y, self, other)

    def __radd__(self, other: Any) -> "MetricsLambda":
        from ignite.metrics.metrics_lambda import MetricsLambda

        return MetricsLambda(lambda x, y: x + y, other, self)

    def __sub__(self, other: Any) -> "MetricsLambda":
        from ignite.metrics.metrics_lambda import MetricsLambda

        return MetricsLambda(lambda x, y: x - y, self, other)

    def __rsub__(self, other: Any) -> "MetricsLambda":
        from ignite.metrics.metrics_lambda import MetricsLambda

        return MetricsLambda(lambda x, y: x - y, other, self)

    def __mul__(self, other: Any) -> "MetricsLambda":
        from ignite.metrics.metrics_lambda import MetricsLambda

        return MetricsLambda(lambda x, y: x * y, self, other)

    def __rmul__(self, other: Any) -> "MetricsLambda":
        from ignite.metrics.metrics_lambda import MetricsLambda

        return MetricsLambda(lambda x, y: x * y, other, self)

    def __pow__(self, other: Any) -> "MetricsLambda":
        from ignite.metrics.metrics_lambda import MetricsLambda

        return MetricsLambda(lambda x, y: x ** y, self, other)

    def __rpow__(self, other: Any) -> "MetricsLambda":
        from ignite.metrics.metrics_lambda import MetricsLambda

        return MetricsLambda(lambda x, y: x ** y, other, self)

    def __mod__(self, other: Any) -> "MetricsLambda":
        from ignite.metrics.metrics_lambda import MetricsLambda

        return MetricsLambda(lambda x, y: x % y, self, other)

    def __div__(self, other: Any) -> "MetricsLambda":
        from ignite.metrics.metrics_lambda import MetricsLambda

        return MetricsLambda(lambda x, y: x.__div__(y), self, other)

    def __rdiv__(self, other: Any) -> "MetricsLambda":
        from ignite.metrics.metrics_lambda import MetricsLambda

        return MetricsLambda(lambda x, y: x.__div__(y), other, self)

    def __truediv__(self, other: Any) -> "MetricsLambda":
        from ignite.metrics.metrics_lambda import MetricsLambda

        return MetricsLambda(lambda x, y: x.__truediv__(y), self, other)

    def __rtruediv__(self, other: Any) -> "MetricsLambda":
        from ignite.metrics.metrics_lambda import MetricsLambda

        return MetricsLambda(lambda x, y: x.__truediv__(y), other, self)

    def __floordiv__(self, other: Any) -> "MetricsLambda":
        from ignite.metrics.metrics_lambda import MetricsLambda

        return MetricsLambda(lambda x, y: x // y, self, other)

    def __getattr__(self, attr: str) -> Callable:
        from ignite.metrics.metrics_lambda import MetricsLambda

        def fn(x: Metric, *args: Any, **kwargs: Any) -> Any:
            return getattr(x, attr)(*args, **kwargs)

        def wrapper(*args: Any, **kwargs: Any) -> "MetricsLambda":
            return MetricsLambda(fn, self, *args, **kwargs)

        return wrapper

    def __getitem__(self, index: Any) -> "MetricsLambda":
        from ignite.metrics.metrics_lambda import MetricsLambda

        return MetricsLambda(lambda x: x[index], self)

    def __getstate__(self) -> Dict:
        return self.__dict__

    def __setstate__(self, d: Dict) -> None:
        self.__dict__.update(d)


def sync_all_reduce(*attrs: Any) -> Callable:
    """Helper decorator for distributed configuration to collect instance attribute value
    across all participating processes.

    See :doc:`metrics` on how to use it.

    Args:
        *attrs: attribute names of decorated class

    """

    def wrapper(func: Callable) -> Callable:
        @wraps(func)
        def another_wrapper(self: Metric, *args: Any, **kwargs: Any) -> Callable:
            if not isinstance(self, Metric):
                raise RuntimeError(
                    "Decorator sync_all_reduce should be used on ignite.metric.Metric class methods only"
                )
            ws = idist.get_world_size()
            if len(attrs) > 0 and not self._is_reduced:
                if ws > 1:
                    for attr in attrs:
                        t = getattr(self, attr, None)
                        if t is not None:
                            t = idist.all_reduce(t)
                            self._is_reduced = True
                            setattr(self, attr, t)
                else:
                    self._is_reduced = True

            return func(self, *args, **kwargs)

        return another_wrapper

    setattr(wrapper, "_decorated", True)
    return wrapper


def reinit__is_reduced(func: Callable) -> Callable:
    """Helper decorator for distributed configuration.

    See :doc:`metrics` on how to use it.

    """

    @wraps(func)
    def wrapper(self: Metric, *args: Any, **kwargs: Any) -> None:
        func(self, *args, **kwargs)
        self._is_reduced = False

    setattr(wrapper, "_decorated", True)
    return wrapper
