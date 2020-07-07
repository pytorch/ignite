import numbers
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Any, Mapping

import torch
from torch.optim import Optimizer

from ignite.engine import Engine, State


class BaseHandler(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, engine, logger, event_name):
        pass


class BaseOptimizerParamsHandler(BaseHandler):
    """
    Base handler for logging optimizer parameters
    """

    def __init__(self, optimizer, param_name="lr", tag=None):
        if not (
            isinstance(optimizer, Optimizer)
            or (hasattr(optimizer, "param_groups") and isinstance(optimizer.param_groups, Sequence))
        ):
            raise TypeError(
                "Argument optimizer should be torch.optim.Optimizer or has attribute 'param_groups' as list/tuple, "
                "but given {}".format(type(optimizer))
            )

        self.optimizer = optimizer
        self.param_name = param_name
        self.tag = tag


class BaseOutputHandler(BaseHandler):
    """
    Helper handler to log engine's output and/or metrics
    """

    def __init__(self, tag, metric_names=None, output_transform=None, global_step_transform=None):

        if metric_names is not None:
            if not (isinstance(metric_names, list) or (isinstance(metric_names, str) and metric_names == "all")):
                raise TypeError(
                    "metric_names should be either a list or equal 'all', " "got {} instead.".format(type(metric_names))
                )

        if output_transform is not None and not callable(output_transform):
            raise TypeError("output_transform should be a function, got {} instead.".format(type(output_transform)))

        if output_transform is None and metric_names is None:
            raise ValueError("Either metric_names or output_transform should be defined")

        if global_step_transform is not None and not callable(global_step_transform):
            raise TypeError(
                "global_step_transform should be a function, got {} instead.".format(type(global_step_transform))
            )

        if global_step_transform is None:

            def global_step_transform(engine, event_name):
                return engine.state.get_event_attrib_value(event_name)

        self.tag = tag
        self.metric_names = metric_names
        self.output_transform = output_transform
        self.global_step_transform = global_step_transform

    def _setup_output_metrics(self, engine):
        """Helper method to setup metrics to log
        """
        metrics = {}
        if self.metric_names is not None:
            if isinstance(self.metric_names, str) and self.metric_names == "all":
                metrics = engine.state.metrics
            else:
                for name in self.metric_names:
                    if name not in engine.state.metrics:
                        warnings.warn(
                            "Provided metric name '{}' is missing "
                            "in engine's state metrics: {}".format(name, list(engine.state.metrics.keys()))
                        )
                        continue
                    metrics[name] = engine.state.metrics[name]

        if self.output_transform is not None:
            output_dict = self.output_transform(engine.state.output)

            if not isinstance(output_dict, dict):
                output_dict = {"output": output_dict}

            metrics.update({name: value for name, value in output_dict.items()})
        return metrics


class BaseWeightsScalarHandler(BaseHandler):
    """
    Helper handler to log model's weights as scalars.
    """

    def __init__(self, model, reduction=torch.norm, tag=None):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Argument model should be of type torch.nn.Module, " "but given {}".format(type(model)))

        if not callable(reduction):
            raise TypeError("Argument reduction should be callable, " "but given {}".format(type(reduction)))

        def _is_0D_tensor(t):
            return isinstance(t, torch.Tensor) and t.ndimension() == 0

        # Test reduction function on a tensor
        o = reduction(torch.ones(4, 2))
        if not (isinstance(o, numbers.Number) or _is_0D_tensor(o)):
            raise TypeError("Output of the reduction function should be a scalar, but got {}".format(type(o)))

        self.model = model
        self.reduction = reduction
        self.tag = tag


class BaseWeightsHistHandler(BaseHandler):
    """
    Helper handler to log model's weights as histograms.
    """

    def __init__(self, model, tag=None):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Argument model should be of type torch.nn.Module, " "but given {}".format(type(model)))

        self.model = model
        self.tag = tag


class BaseLogger(metaclass=ABCMeta):
    """
    Base logger handler. See implementations: TensorboardLogger, VisdomLogger, PolyaxonLogger, MLflowLogger, ...

    """

    def attach(self, engine, log_handler, event_name):
        """Attach the logger to the engine and execute `log_handler` function at `event_name` events.

        Args:
            engine (Engine): engine object.
            log_handler (callable): a logging handler to execute
            event_name: event to attach the logging handler to. Valid events are from
                :class:`~ignite.engine.events.Events` or any `event_name` added by
                :meth:`~ignite.engine.engine.Engine.register_events`.

        Returns:
            :class:`~ignite.engine.RemovableEventHandle`, which can be used to remove the handler.
        """
        name = event_name

        if name not in State.event_to_attr:
            raise RuntimeError("Unknown event name '{}'".format(name))

        return engine.add_event_handler(event_name, log_handler, self, name)

    def attach_output_handler(self, engine: Engine, event_name: Any, *args: Any, **kwargs: Mapping):
        """Shortcut method to attach `OutputHandler` to the logger.

        Args:
            engine (Engine): engine object.
            event_name: event to attach the logging handler to. Valid events are from
                :class:`~ignite.engine.events.Events` or any `event_name` added by
                :meth:`~ignite.engine.engine.Engine.register_events`.
            *args: args to initialize `OutputHandler`
            **kwargs: kwargs to initialize `OutputHandler`

        Returns:
            :class:`~ignite.engine.RemovableEventHandle`, which can be used to remove the handler.
        """
        return self.attach(engine, self._create_output_handler(*args, **kwargs), event_name=event_name)

    def attach_opt_params_handler(self, engine: Engine, event_name: Any, *args: Any, **kwargs: Mapping):
        """Shortcut method to attach `OptimizerParamsHandler` to the logger.

        Args:
            engine (Engine): engine object.
            event_name: event to attach the logging handler to. Valid events are from
                :class:`~ignite.engine.events.Events` or any `event_name` added by
                :meth:`~ignite.engine.engine.Engine.register_events`.
            *args: args to initialize `OptimizerParamsHandler`
            **kwargs: kwargs to initialize `OptimizerParamsHandler`

        Returns:
            :class:`~ignite.engine.RemovableEventHandle`, which can be used to remove the handler.
        """
        self.attach(engine, self._create_opt_params_handler(*args, **kwargs), event_name=event_name)

    @abstractmethod
    def _create_output_handler(self, engine, *args, **kwargs):
        pass

    @abstractmethod
    def _create_opt_params_handler(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        pass
