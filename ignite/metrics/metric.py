from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping
from functools import wraps
from numbers import Number
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import torch

import ignite.distributed as idist

from ignite.base.mixins import Serializable
from ignite.engine import CallableEventWithFilter, Engine, Events
from ignite.utils import _CollectionItem, _tree_apply2, _tree_map

if TYPE_CHECKING:
    from ignite.metrics.metrics_lambda import MetricsLambda

__all__ = [
    "Metric",
    "MetricUsage",
    "EpochWise",
    "BatchWise",
    "BatchFiltered",
    "RunningEpochWise",
    "RunningBatchWise",
    "SingleEpochRunningBatchWise",
]


class MetricUsage:
    """
    Base class for all usages of metrics.

    A usage of metric defines the events when a metric starts to compute, updates and completes.
    Valid events are from :class:`~ignite.engine.events.Events`.

    Args:
        started: event when the metric starts to compute. This event will be associated to
            :meth:`~ignite.metrics.metric.Metric.started`.
        completed: event when the metric completes. This event will be associated to
            :meth:`~ignite.metrics.metric.Metric.completed`.
        iteration_completed: event when the metric updates. This event will be associated to
            :meth:`~ignite.metrics.metric.Metric.iteration_completed`.
    """

    usage_name: str

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

    - :meth:`~ignite.metrics.metric.Metric.started` on every ``EPOCH_STARTED``
      (See :class:`~ignite.engine.events.Events`).
    - :meth:`~ignite.metrics.metric.Metric.iteration_completed` on every ``ITERATION_COMPLETED``.
    - :meth:`~ignite.metrics.metric.Metric.completed` on every ``EPOCH_COMPLETED``.

    Attributes:
        usage_name: usage name string
    """

    usage_name: str = "epoch_wise"

    def __init__(self) -> None:
        super(EpochWise, self).__init__(
            started=Events.EPOCH_STARTED,
            completed=Events.EPOCH_COMPLETED,
            iteration_completed=Events.ITERATION_COMPLETED,
        )


class RunningEpochWise(EpochWise):
    """
    Running epoch-wise usage of Metrics. It's the running version of the :class:`~.metrics.metric.EpochWise` metric
    usage. A metric with such a usage most likely accompanies an :class:`~.metrics.metric.EpochWise` one to compute
    a running measure of it e.g. running average.

    Metric's methods are triggered on the following engine events:

    - :meth:`~ignite.metrics.metric.Metric.started` on every ``STARTED``
      (See :class:`~ignite.engine.events.Events`).
    - :meth:`~ignite.metrics.metric.Metric.iteration_completed` on every ``EPOCH_COMPLETED``.
    - :meth:`~ignite.metrics.metric.Metric.completed` on every ``EPOCH_COMPLETED``.

    Attributes:
        usage_name: usage name string
    """

    usage_name: str = "running_epoch_wise"

    def __init__(self) -> None:
        super(EpochWise, self).__init__(
            started=Events.STARTED,
            completed=Events.EPOCH_COMPLETED,
            iteration_completed=Events.EPOCH_COMPLETED,
        )


class BatchWise(MetricUsage):
    """
    Batch-wise usage of Metrics.

    Metric's methods are triggered on the following engine events:

    - :meth:`~ignite.metrics.metric.Metric.started` on every ``ITERATION_STARTED``
      (See :class:`~ignite.engine.events.Events`).
    - :meth:`~ignite.metrics.metric.Metric.iteration_completed` on every ``ITERATION_COMPLETED``.
    - :meth:`~ignite.metrics.metric.Metric.completed` on every ``ITERATION_COMPLETED``.

    Attributes:
        usage_name: usage name string
    """

    usage_name: str = "batch_wise"

    def __init__(self) -> None:
        super(BatchWise, self).__init__(
            started=Events.ITERATION_STARTED,
            completed=Events.ITERATION_COMPLETED,
            iteration_completed=Events.ITERATION_COMPLETED,
        )


class RunningBatchWise(BatchWise):
    """
    Running batch-wise usage of Metrics. It's the running version of the :class:`~.metrics.metric.EpochWise` metric
    usage. A metric with such a usage could for example accompany a :class:`~.metrics.metric.BatchWise` one to compute
    a running measure of it e.g. running average.

    Metric's methods are triggered on the following engine events:

    - :meth:`~ignite.metrics.metric.Metric.started` on every ``STARTED``
      (See :class:`~ignite.engine.events.Events`).
    - :meth:`~ignite.metrics.metric.Metric.iteration_completed` on every ``ITERATION_COMPLETED``.
    - :meth:`~ignite.metrics.metric.Metric.completed` on every ``ITERATION_COMPLETED``.

    Attributes:
        usage_name: usage name string
    """

    usage_name: str = "running_batch_wise"

    def __init__(self) -> None:
        super(BatchWise, self).__init__(
            started=Events.STARTED,
            completed=Events.ITERATION_COMPLETED,
            iteration_completed=Events.ITERATION_COMPLETED,
        )


class SingleEpochRunningBatchWise(BatchWise):
    """
    Running batch-wise usage of Metrics in a single epoch. It's like :class:`~.metrics.metric.RunningBatchWise` metric
    usage with the difference that is used during a single epoch.

    Metric's methods are triggered on the following engine events:

    - :meth:`~ignite.metrics.metric.Metric.started` on every ``EPOCH_STARTED``
      (See :class:`~ignite.engine.events.Events`).
    - :meth:`~ignite.metrics.metric.Metric.iteration_completed` on every ``ITERATION_COMPLETED``.
    - :meth:`~ignite.metrics.metric.Metric.completed` on every ``ITERATION_COMPLETED``.

    Attributes:
        usage_name: usage name string
    """

    usage_name: str = "single_epoch_running_batch_wise"

    def __init__(self) -> None:
        super(BatchWise, self).__init__(
            started=Events.EPOCH_STARTED,
            completed=Events.ITERATION_COMPLETED,
            iteration_completed=Events.ITERATION_COMPLETED,
        )


class BatchFiltered(MetricUsage):
    """
    Batch filtered usage of Metrics. This usage is similar to epoch-wise but update event is filtered.

    Metric's methods are triggered on the following engine events:

    - :meth:`~ignite.metrics.metric.Metric.started` on every ``EPOCH_STARTED``
      (See :class:`~ignite.engine.events.Events`).
    - :meth:`~ignite.metrics.metric.Metric.iteration_completed` on filtered ``ITERATION_COMPLETED``.
    - :meth:`~ignite.metrics.metric.Metric.completed` on every ``EPOCH_COMPLETED``.

    Args:
        args: Positional arguments to setup :attr:`~ignite.engine.events.Events.ITERATION_COMPLETED`
        kwargs: Keyword arguments to setup :attr:`~ignite.engine.events.Events.ITERATION_COMPLETED`
            handled by :meth:`~ignite.metrics.metric.Metric.iteration_completed`.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(BatchFiltered, self).__init__(
            started=Events.EPOCH_STARTED,
            completed=Events.EPOCH_COMPLETED,
            iteration_completed=Events.ITERATION_COMPLETED(*args, **kwargs),
        )


class Metric(Serializable, metaclass=ABCMeta):
    """
    Base class for all Metrics.

    Args:
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
        skip_unrolling: specifies whether output should be unrolled before being fed to update method. Should be
            true for multi-output model, for example, if ``y_pred`` contains multi-ouput as ``(y_pred_a, y_pred_b)``
            Alternatively, ``output_transform`` can be used to handle this.

            Examples:
                The following example shows a custom loss metric that expects input from a multi-output model.

                .. code-block:: python

                    import torch
                    import torch.nn as nn
                    import torch.nn.functional as F

                    from ignite.engine import create_supervised_evaluator
                    from ignite.metrics import Loss

                    class MyLoss(nn.Module):
                        def __init__(self, ca: float = 1.0, cb: float = 1.0) -> None:
                            super().__init__()
                            self.ca = ca
                            self.cb = cb

                        def forward(self,
                                    y_pred: Tuple[torch.Tensor, torch.Tensor],
                                    y_true: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
                            a_true, b_true = y_true
                            a_pred, b_pred = y_pred
                            return self.ca * F.mse_loss(a_pred, a_true) + self.cb * F.cross_entropy(b_pred, b_true)


                    def prepare_batch(batch, device, non_blocking):
                        return torch.rand(4, 1), (torch.rand(4, 1), torch.rand(4, 2))


                    class MyModel(nn.Module):

                        def forward(self, x):
                            return torch.rand(4, 1), torch.rand(4, 2)


                    model = MyModel()

                    device = "cpu"
                    loss = MyLoss(0.5, 1.0)
                    metrics = {
                        "Loss": Loss(loss, skip_unrolling=True)
                    }
                    train_evaluator = create_supervised_evaluator(model, metrics, device, prepare_batch=prepare_batch)


                    data = range(10)
                    train_evaluator.run(data)
                    train_evaluator.state.metrics["Loss"]

    Attributes:
        required_output_keys: dictionary defines required keys to be found in ``engine.state.output`` if the
            latter is a dictionary. By default, ``("y_pred", "y")``. This is useful with custom metrics that can require
            other arguments than predictions ``y_pred`` and targets ``y``. See an example below.

    Examples:
        Let's implement a custom metric that requires ``y_pred``, ``y`` and ``x`` as input for ``update`` function.
        In the example below we show how to setup standard metric like Accuracy and the custom metric using by an
        ``evaluator`` created with :meth:`~ignite.engine.create_supervised_evaluator` method.

        For more information on how metric works with :class:`~ignite.engine.engine.Engine`, visit :ref:`attach-engine`.

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

    .. versionchanged:: 0.4.2
        ``required_output_keys`` became public attribute.

    .. versionchanged:: 0.5.1
        ``skip_unrolling`` argument is added.
    """

    # public class attribute
    required_output_keys: Optional[Tuple] = ("y_pred", "y")
    # for backward compatibility
    _required_output_keys = required_output_keys

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        self._output_transform = output_transform

        # Some metrics have a large performance regression when run on XLA devices, so for now, we disallow it.
        if torch.device(device).type == "xla":
            raise ValueError("Cannot create metric on an XLA device. Use device='cpu' instead.")

        self._device = torch.device(device)
        self._skip_unrolling = skip_unrolling
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the metric to its initial state.

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
        Computes the metric based on its accumulated state.

        By default, this is called at the end of each epoch.

        Returns:
            Any: | the actual quantity of interest. However, if a :class:`~collections.abc.Mapping` is returned,
                 it will be (shallow) flattened into `engine.state.metrics` when
                 :func:`~ignite.metrics.metric.Metric.completed` is called.

        Raises:
            NotComputableError: raised when the metric cannot be computed.
        """
        pass

    def started(self, engine: Engine) -> None:
        """Helper method to start data gathering for metric's computation. It is automatically attached to the
        `engine` with :meth:`~ignite.metrics.metric.Metric.attach`.

        Args:
            engine: the engine to which the metric must be attached
        """
        self.reset()

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        """Helper method to update metric's computation. It is automatically attached to the
        `engine` with :meth:`~ignite.metrics.metric.Metric.attach`.

        Args:
            engine: the engine to which the metric must be attached

        Note:
            ``engine.state.output`` is used to compute metric values.
            The majority of implemented metrics accept the following formats for ``engine.state.output``:
            ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``. ``y_pred`` and ``y`` can be torch tensors or
            list of tensors/numbers if applicable.

        .. versionchanged:: 0.4.5
            ``y_pred`` and ``y`` can be torch tensors or list of tensors/numbers
        """

        output = self._output_transform(engine.state.output)
        if isinstance(output, Mapping):
            if self.required_output_keys is None:
                raise TypeError(
                    f"Transformed engine output for {self.__class__.__name__} metric should be a tuple/list, "
                    f"but given {type(output)}"
                )
            if not all([k in output for k in self.required_output_keys]):
                raise ValueError(
                    "When transformed engine's output is a mapping, "
                    f"it should contain {self.required_output_keys} keys, but given {list(output.keys())}"
                )
            output = tuple(output[k] for k in self.required_output_keys)

        if (
            (not self._skip_unrolling)
            and isinstance(output, Sequence)
            and all([_is_list_of_tensors_or_numbers(o) for o in output])
        ):
            if not (len(output) == 2 and len(output[0]) == len(output[1])):
                raise ValueError(
                    f"Output should have 2 items of the same length, "
                    f"got {len(output)} and {len(output[0])}, {len(output[1])}"
                )
            for o1, o2 in zip(output[0], output[1]):
                # o1 and o2 are list of tensors or numbers
                tensor_o1 = _to_batched_tensor(o1)
                tensor_o2 = _to_batched_tensor(o2, device=tensor_o1.device)
                self.update((tensor_o1, tensor_o2))
        else:
            self.update(output)

    def completed(self, engine: Engine, name: str) -> None:
        """Helper method to compute metric's value and put into the engine. It is automatically attached to the
        `engine` with :meth:`~ignite.metrics.metric.Metric.attach`. If metrics' value is torch tensor, it is
        explicitly sent to CPU device.

        Args:
            engine: the engine to which the metric must be attached
            name: the name of the metric used as key in dict `engine.state.metrics`

        .. versionchanged:: 0.4.3
            Added dict in metrics results.

        .. versionchanged:: 0.4.5
            metric's value is put on CPU if torch tensor.

        """
        result = self.compute()
        if isinstance(result, Mapping):
            if name in result.keys():
                raise ValueError(f"Argument name '{name}' is conflicting with mapping keys: {list(result.keys())}")

            for key, value in result.items():
                engine.state.metrics[key] = value
            engine.state.metrics[name] = result
        else:
            if isinstance(result, torch.Tensor):
                if len(result.size()) == 0:
                    result = result.item()
                elif "cpu" not in result.device.type:
                    result = result.cpu()

            engine.state.metrics[name] = result

    def _check_usage(self, usage: Union[str, MetricUsage]) -> MetricUsage:
        if isinstance(usage, str):
            usages = [EpochWise, RunningEpochWise, BatchWise, RunningBatchWise, SingleEpochRunningBatchWise]
            for usage_cls in usages:
                if usage == usage_cls.usage_name:
                    usage = usage_cls()
                    break
            if not isinstance(usage, MetricUsage):
                raise ValueError(
                    "Argument usage should be '(Running)EpochWise.usage_name' or "
                    f"'((SingleEpoch)Running)BatchWise.usage_name', got {usage}"
                )
        if not isinstance(usage, MetricUsage):
            raise TypeError(f"Unhandled usage type {type(usage)}")
        return usage

    def attach(self, engine: Engine, name: str, usage: Union[str, MetricUsage] = EpochWise()) -> None:
        """
        Attaches current metric to provided engine. On the end of engine's run, `engine.state.metrics` dictionary will
        contain computed metric's value under provided name.

        Args:
            engine: the engine to which the metric must be attached
            name: the name of the metric to attach
            usage: the usage of the metric. Valid string values should be
                :attr:`ignite.metrics.metric.EpochWise.usage_name` (default) or
                :attr:`ignite.metrics.metric.BatchWise.usage_name`.

        Examples:

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
        This method in conjunction with :meth:`~ignite.metrics.metric.Metric.attach` can be useful if several
        metrics need to be computed with different periods. For example, one metric is computed every training epoch
        and another metric (e.g. more expensive one) is done every n-th training epoch.

        Args:
            engine: the engine from which the metric must be detached
            usage: the usage of the metric. Valid string values should be
                'epoch_wise' (default) or 'batch_wise'.

        Examples:
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
            engine: the engine checked from which the metric should be attached
            usage: the usage of the metric. Valid string values should be
                'epoch_wise' (default) or 'batch_wise'.
        """
        usage = self._check_usage(usage)
        return engine.has_event_handler(self.completed, usage.COMPLETED)

    def _state_dict_per_rank(self) -> OrderedDict:
        def func(
            x: Union[torch.Tensor, Metric, None, float], **kwargs: Any
        ) -> Union[torch.Tensor, float, OrderedDict, None]:
            if isinstance(x, Metric):
                return x._state_dict_per_rank()
            if x is None or isinstance(x, (int, float, torch.Tensor)):
                return x
            else:
                raise TypeError(
                    "Found attribute of unsupported type. Currently, supported types include"
                    " numeric types, tensor, Metric or sequence/mapping of metrics."
                )

        state: OrderedDict[str, Union[torch.Tensor, List, Dict, None]] = OrderedDict()
        for attr_name in self._state_dict_all_req_keys:
            if attr_name not in self.__dict__:
                raise ValueError(
                    f"Found a value in _state_dict_all_req_keys that is not among metric attributes: {attr_name}"
                )
            attr = getattr(self, attr_name)
            state[attr_name] = _tree_map(func, attr)  # type: ignore[assignment]

        return state

    __state_dict_key_per_rank: str = "__metric_state_per_rank"

    def state_dict(self) -> OrderedDict:
        """Method returns state dict with attributes of the metric specified in its
        `_state_dict_all_req_keys` attribute. Can be used to save internal state of the class.
        """
        state = self._state_dict_per_rank()

        if idist.get_world_size() > 1:
            return OrderedDict([(Metric.__state_dict_key_per_rank, idist.all_gather(state))])
        return OrderedDict([(Metric.__state_dict_key_per_rank, [state])])

    def _load_state_dict_per_rank(self, state_dict: Mapping) -> None:
        super().load_state_dict(state_dict)

        def func(x: Any, y: Any) -> None:
            if isinstance(x, Metric):
                x._load_state_dict_per_rank(y)
            elif isinstance(x, _CollectionItem):
                value = x.value()
                if y is None or isinstance(y, _CollectionItem.types_as_collection_item):
                    x.load_value(y)
                elif isinstance(value, Metric):
                    value._load_state_dict_per_rank(y)
                else:
                    raise ValueError(f"Unsupported type for provided state_dict data: {type(y)}")

        for attr_name in self._state_dict_all_req_keys:
            attr = getattr(self, attr_name)
            attr = _CollectionItem.wrap(self.__dict__, attr_name, attr)
            _tree_apply2(func, attr, state_dict[attr_name])

    def load_state_dict(self, state_dict: Mapping) -> None:
        """Method replaces internal state of the class with provided state dict data.

        If there's an active distributed configuration, the process uses its rank to pick the proper value from
        the list of values saved under each attribute's name in the dict.

        Args:
            state_dict: a dict containing attributes of the metric specified in its `_state_dict_all_req_keys`
                attribute.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Argument state_dict should be a dictionary, but given {type(state_dict)}")

        if not (len(state_dict) == 1 and Metric.__state_dict_key_per_rank in state_dict):
            raise ValueError(
                "Incorrect state_dict object. Argument state_dict should be a dictionary "
                "provided by Metric.state_dict(). "
                f"Expected single key: {Metric.__state_dict_key_per_rank}, but given {state_dict.keys()}"
            )

        list_state_dicts_per_rank = state_dict[Metric.__state_dict_key_per_rank]
        rank = idist.get_rank()
        world_size = idist.get_world_size()
        if len(list_state_dicts_per_rank) != world_size:
            raise ValueError(
                "Incorrect state_dict object. Argument state_dict should be a dictionary "
                "provided by Metric.state_dict(). "
                f"Expected a list of state_dicts of size equal world_size: {world_size}, "
                f"but got {len(list_state_dicts_per_rank)}"
            )

        state_dict = list_state_dicts_per_rank[rank]
        self._load_state_dict_per_rank(state_dict)

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

        return MetricsLambda(lambda x, y: x**y, self, other)

    def __rpow__(self, other: Any) -> "MetricsLambda":
        from ignite.metrics.metrics_lambda import MetricsLambda

        return MetricsLambda(lambda x, y: x**y, other, self)

    def __mod__(self, other: Any) -> "MetricsLambda":
        from ignite.metrics.metrics_lambda import MetricsLambda

        return MetricsLambda(lambda x, y: x % y, self, other)

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

        if attr.startswith("__") and attr.endswith("__"):
            return object.__getattribute__(self, attr)

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
    across all participating processes and apply the specified reduction operation.

    See :doc:`metrics` on how to use it.

    Args:
        attrs: attribute names of decorated class

    .. versionchanged:: 0.4.5
        - Ability to handle different reduction operations (SUM, MAX, MIN, PRODUCT).
    """

    def wrapper(func: Callable) -> Callable:
        @wraps(func)
        def another_wrapper(self: Metric, *args: Any, **kwargs: Any) -> Callable:
            if not isinstance(self, Metric):
                raise RuntimeError(
                    "Decorator sync_all_reduce should be used on ignite.metric.Metric class methods only"
                )
            ws = idist.get_world_size()
            unreduced_attrs = {}
            if len(attrs) > 0 and ws > 1:
                for attr in attrs:
                    op_kwargs = {}
                    if ":" in attr:
                        attr, op = attr.split(":")
                        valid_ops = ["MIN", "MAX", "SUM", "PRODUCT"]
                        if op not in valid_ops:
                            raise ValueError(f"Reduction operation is not valid (expected : {valid_ops}, got: {op}")
                        op_kwargs["op"] = op
                    if attr not in self.__dict__:
                        raise ValueError(f"Metric {type(self)} has no attribute named `{attr}`.")
                    t = getattr(self, attr)
                    if not isinstance(t, (Number, torch.Tensor)):
                        raise TypeError(
                            "Attribute provided to sync_all_reduce should be a "
                            f"number or tensor but `{attr}` has type {type(t)}"
                        )
                    unreduced_attrs[attr] = t
                    # Here `clone` is necessary since `idist.all_reduce` modifies `t` inplace in the case
                    # `t` is a tensor and its `device` is same as that of the process.
                    # TODO: Remove this dual behavior of `all_reduce` to always either return a new tensor or
                    #       modify it in-place.
                    t_reduced = idist.all_reduce(cast(float, t) if isinstance(t, Number) else t.clone(), **op_kwargs)
                    setattr(self, attr, t_reduced)

            result = func(self, *args, **kwargs)

            for attr, value in unreduced_attrs.items():
                setattr(self, attr, value)
            return result

        return another_wrapper

    setattr(wrapper, "_decorated", True)
    return wrapper


def reinit__is_reduced(func: Callable) -> Callable:
    """Helper decorator for distributed configuration.

    See :doc:`metrics` on how to use it.

    Args:
        func: A callable to reinit.
    """

    @wraps(func)
    def wrapper(self: Metric, *args: Any, **kwargs: Any) -> None:
        func(self, *args, **kwargs)
        if "_result" in self.__dict__:
            self._result = None  # type: ignore[attr-defined]

    setattr(wrapper, "_decorated", True)
    return wrapper


def _is_list_of_tensors_or_numbers(x: Sequence[Union[torch.Tensor, float]]) -> bool:
    return isinstance(x, Sequence) and all([isinstance(t, (torch.Tensor, Number)) for t in x])


def _to_batched_tensor(x: Union[torch.Tensor, float], device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.unsqueeze(dim=0)
    return torch.tensor([x], device=device)
