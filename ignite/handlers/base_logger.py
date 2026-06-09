"""Base logger and its helper handlers."""

import numbers
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ignite.engine import Engine, Events, EventsList, State
from ignite.engine.events import CallableEventWithFilter, RemovableEventHandle


class BaseHandler(ABC):
    """Base handler for defining various useful handlers."""

    @abstractmethod
    def __call__(self, engine: Engine, logger: Any, event_name: str | Events) -> None: ...


class BaseWeightsHandler(BaseHandler):
    """
    Base handler for logging weights or their gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        tag: str | None = None,
        whitelist: list[str] | Callable[[str, nn.Parameter], bool] | None = None,
    ):
        if not isinstance(model, nn.Module):
            raise TypeError(f"Argument model should be of type torch.nn.Module, but given {type(model)}")

        self.model = model
        self.tag = tag

        if whitelist is None:
            weights = dict(model.named_parameters())
        elif callable(whitelist):
            weights = {name: param for name, param in model.named_parameters() if whitelist(name, param)}
        else:
            prefixes = tuple(whitelist)
            weights = {name: param for name, param in model.named_parameters() if name.startswith(prefixes)}

        self.weights = tuple(weights.items())


class BaseOptimizerParamsHandler(BaseHandler):
    """
    Base handler for logging optimizer parameters
    """

    def __init__(self, optimizer: Optimizer, param_name: str = "lr", tag: str | None = None):
        if not (
            isinstance(optimizer, Optimizer)
            or (hasattr(optimizer, "param_groups") and isinstance(optimizer.param_groups, Sequence))
        ):
            raise TypeError(
                "Argument optimizer should be torch.optim.Optimizer or has attribute 'param_groups' as list/tuple, "
                f"but given {type(optimizer)}"
            )

        self.optimizer = optimizer
        self.param_name = param_name
        self.tag = tag


class BaseOutputHandler(BaseHandler):
    """
    Helper handler to log engine's output and/or metrics
    """

    def __init__(
        self,
        tag: str,
        metric_names: str | list[str] | None = None,
        output_transform: Callable | None = None,
        global_step_transform: Callable[[Engine, str | Events], int] | None = None,
        state_attributes: list[str] | None = None,
    ):
        if metric_names is not None:
            if not (isinstance(metric_names, list) or metric_names == "all"):
                raise TypeError(
                    f"metric_names should be either a list or equal 'all', got {type(metric_names)} instead."
                )

        if output_transform is not None and not callable(output_transform):
            raise TypeError(f"output_transform should be a function, got {type(output_transform)} instead.")

        if output_transform is None and metric_names is None and state_attributes is None:
            raise ValueError("Either metric_names, output_transform or state_attributes should be defined")

        if global_step_transform is not None and not callable(global_step_transform):
            raise TypeError(f"global_step_transform should be a function, got {type(global_step_transform)} instead.")

        if global_step_transform is None:

            def global_step_transform(engine: Engine, event_name: str | Events) -> int:
                return engine.state.get_event_attrib_value(event_name)

        self.tag = tag
        self.metric_names = metric_names
        self.output_transform = output_transform
        self.global_step_transform = global_step_transform
        self.state_attributes = state_attributes

    def _setup_output_metrics_state_attrs(
        self, engine: Engine, log_text: bool | None = False, key_tuple: bool | None = True
    ) -> dict[Any, Any]:
        """Helper method to setup metrics and state attributes to log"""
        metrics_state_attrs = OrderedDict()
        if self.metric_names is not None:
            if isinstance(self.metric_names, str) and self.metric_names == "all":
                metrics_state_attrs = OrderedDict(engine.state.metrics)
            else:
                for name in self.metric_names:
                    if name not in engine.state.metrics:
                        warnings.warn(
                            f"Provided metric name '{name}' is missing "
                            f"in engine's state metrics: {list(engine.state.metrics.keys())}"
                        )
                        continue
                    metrics_state_attrs[name] = engine.state.metrics[name]

        if self.output_transform is not None:
            output_dict = self.output_transform(engine.state.output)

            if not isinstance(output_dict, dict):
                output_dict = {"output": output_dict}

            metrics_state_attrs.update(output_dict)

        if self.state_attributes is not None:
            metrics_state_attrs.update({name: getattr(engine.state, name, None) for name in self.state_attributes})

        def key_tuple_fn(parent_key: str | tuple[str, ...] | None, *args: str) -> tuple[str, ...]:
            if parent_key is None:
                return args
            if isinstance(parent_key, str):
                return (parent_key,) + args
            return parent_key + args

        def key_str_fn(parent_key: str, *args: str) -> str:
            args_str = "/".join(args)
            return f"{parent_key}/{args_str}"

        key_fn = key_tuple_fn if key_tuple else key_str_fn

        def handle_value_fn(
            value: str | int | float | numbers.Number | torch.Tensor,
        ) -> None | str | float | numbers.Number:
            if isinstance(value, numbers.Number):
                return value

            if isinstance(value, torch.Tensor) and value.ndimension() == 0:
                return value.item()

            if isinstance(value, str) and log_text:
                return value

            warnings.warn(f"Logger output_handler can not log metrics value type {type(value)}")
            return None

        metrics_state_attrs_dict = _flatten_dict(metrics_state_attrs, key_fn, handle_value_fn, parent_key=self.tag)
        return metrics_state_attrs_dict


def _flatten_dict(
    in_dict: Mapping,
    key_fn: Callable,
    value_fn: Callable,
    parent_key: str | tuple[str, ...] | None = None,
) -> dict:
    """Recursively flatten a nested mapping.

    Args:
        in_dict: mapping to flatten.
        key_fn: function used to build the flattened key from the parent key and current key.
        value_fn: function used to convert leaf values to loggable values.
        parent_key: parent key prefix for nested values.

    Returns:
        Flattened dictionary of processed key-value pairs.
    """
    items = {}
    for key, value in in_dict.items():
        new_key = key_fn(parent_key, key)
        if isinstance(value, Mapping):
            items.update(_flatten_dict(value, key_fn, value_fn, new_key))
        elif any(
            [
                isinstance(value, tuple) and hasattr(value, "_fields"),  # namedtuple
                not isinstance(value, str) and isinstance(value, Sequence),
            ]
        ):
            for i, item in enumerate(value):
                items.update(_flatten_dict({str(i): item}, key_fn, value_fn, new_key))
        elif isinstance(value, torch.Tensor) and value.ndimension() == 1:
            for i, item in enumerate(value):
                items.update(_flatten_dict({str(i): item.item()}, key_fn, value_fn, new_key))
        else:
            new_value = value_fn(value)
            if new_value is not None:
                items[new_key] = new_value
    return items


class BaseWeightsScalarHandler(BaseWeightsHandler):
    """Helper handler to log model's weights or gradients as scalars."""

    def __init__(
        self,
        model: nn.Module,
        reduction: Callable[[torch.Tensor], float | torch.Tensor] = torch.norm,
        tag: str | None = None,
        whitelist: list[str] | Callable[[str, nn.Parameter], bool] | None = None,
    ):
        super().__init__(model, tag=tag, whitelist=whitelist)

        if not callable(reduction):
            raise TypeError(f"Argument reduction should be callable, but given {type(reduction)}")

        def _is_0d_tensor(t: Any) -> bool:
            return isinstance(t, torch.Tensor) and t.ndimension() == 0

        # Test reduction function on a tensor
        o = reduction(torch.ones(4, 2))
        if not (isinstance(o, numbers.Number) or _is_0d_tensor(o)):
            raise TypeError(f"Output of the reduction function should be a scalar, but got {type(o)}")

        self.reduction = reduction


class BaseLogger(ABC):
    """
    Base logger handler. See implementations: TensorboardLogger, VisdomLogger, PolyaxonLogger, MLflowLogger, ...

    """

    def attach(
        self,
        engine: Engine,
        log_handler: Callable,
        event_name: str | Events | CallableEventWithFilter | EventsList,
        *args: Any,
        **kwargs: Any,
    ) -> RemovableEventHandle:
        """Attach the logger to the engine and execute `log_handler` at `event_name` events.

        Args:
            engine: engine object.
            log_handler: logging handler to execute.
            event_name: event to attach the logging handler to. Valid events are from
                :class:`~ignite.engine.events.Events` or :class:`~ignite.engine.events.EventsList` or any `event_name`
                added by :meth:`~ignite.engine.engine.Engine.register_events`.
            args: positional arguments forwarded to `log_handler`.
            kwargs: keyword arguments forwarded to `log_handler`.

        Returns:
            :class:`~ignite.engine.events.RemovableEventHandle` that can be used to remove the handler.
        """
        if isinstance(event_name, EventsList):
            for name in event_name:
                if name not in State.event_to_attr:
                    raise RuntimeError(f"Unknown event name '{name}'")
                engine.add_event_handler(name, log_handler, self, name)

            return RemovableEventHandle(event_name, log_handler, engine)

        if event_name not in State.event_to_attr:
            raise RuntimeError(f"Unknown event name '{event_name}'")

        return engine.add_event_handler(event_name, log_handler, self, event_name, *args, **kwargs)

    def attach_output_handler(self, engine: Engine, event_name: Any, *args: Any, **kwargs: Any) -> RemovableEventHandle:
        """Shortcut to attach `OutputHandler` to the logger.

        Args:
            engine: engine object.
            event_name: event to attach the logging handler to. Valid events are from
                :class:`~ignite.engine.events.Events` or any `event_name` added by
                :meth:`~ignite.engine.engine.Engine.register_events`.
            args: positional args for `OutputHandler`.
            kwargs: keyword args for `OutputHandler`.

        Returns:
            :class:`~ignite.engine.events.RemovableEventHandle` that can be used to remove the handler.
        """
        return self.attach(engine, self._create_output_handler(*args, **kwargs), event_name=event_name)

    def attach_opt_params_handler(
        self, engine: Engine, event_name: Any, *args: Any, **kwargs: Any
    ) -> RemovableEventHandle:
        """Shortcut to attach `OptimizerParamsHandler` to the logger.

        Args:
            engine: engine object.
            event_name: event to attach the logging handler to. Valid events are from
                :class:`~ignite.engine.events.Events` or any `event_name` added by
                :meth:`~ignite.engine.engine.Engine.register_events`.
            args: positional args for `OptimizerParamsHandler`.
            kwargs: keyword args for `OptimizerParamsHandler`.

        Returns:
            :class:`~ignite.engine.events.RemovableEventHandle` that can be used to remove the handler.

        .. versionchanged:: 0.4.3
            Added missing return statement.
        """
        return self.attach(engine, self._create_opt_params_handler(*args, **kwargs), event_name=event_name)

    @abstractmethod
    def _create_output_handler(self, engine: Engine, *args: Any, **kwargs: Any) -> Callable: ...

    @abstractmethod
    def _create_opt_params_handler(self, *args: Any, **kwargs: Any) -> Callable: ...

    def __enter__(self) -> "BaseLogger":
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None: ...
