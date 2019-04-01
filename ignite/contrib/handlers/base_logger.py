from abc import ABCMeta, abstractmethod
import numbers
import warnings

import torch

from ignite.engine import State, Engine
from ignite._six import with_metaclass


class BaseLogger(object):
    """
    Base logger handler. See implementations: TensorboardLogger, VisdomLogger, PolyaxonLogger

    """
    def attach(self, engine, log_handler, event_name):
        """Attach the logger to the engine and execute `log_handler` function at `event_name` events.

        Args:
            engine (Engine): engine object.
            log_handler (callable): a logging handler to execute
            event_name: event to attach the logging handler to. Valid events are from :class:`~ignite.engine.Events`
                or any `event_name` added by :meth:`~ignite.engine.Engine.register_events`.

        """
        if event_name not in State.event_to_attr:
            raise RuntimeError("Unknown event name '{}'".format(event_name))

        engine.add_event_handler(event_name, log_handler, self, event_name)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        pass


class BaseHandler(with_metaclass(ABCMeta, object)):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class BaseOptimizerParamsHandler(BaseHandler):
    """
    Base handler for logging optimizer parameters
    """

    def __init__(self, optimizer, param_name="lr"):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("Argument optimizer should be of type torch.optim.Optimizer, "
                            "but given {}".format(type(optimizer)))

        self.optimizer = optimizer
        self.param_name = param_name


class BaseOutputHandler(BaseHandler):
    """
    Helper handler to log engine's output and/or metrics
    """

    def __init__(self, tag, metric_names=None, output_transform=None, another_engine=None):

        if metric_names is not None and not isinstance(metric_names, list):
            raise TypeError("metric_names should be a list, got {} instead.".format(type(metric_names)))

        if output_transform is not None and not callable(output_transform):
            raise TypeError("output_transform should be a function, got {} instead."
                            .format(type(output_transform)))

        if output_transform is None and metric_names is None:
            raise ValueError("Either metric_names or output_transform should be defined")

        if another_engine is not None:
            if not isinstance(another_engine, Engine):
                raise TypeError("Argument another_engine should be of type Engine, "
                                "but given {}".format(type(another_engine)))
        self.tag = tag
        self.metric_names = metric_names
        self.output_transform = output_transform
        self.another_engine = another_engine

    def _setup_output_metrics(self, engine):
        """Helper method to setup metrics to log
        """
        metrics = {}
        if self.metric_names is not None:
            for name in self.metric_names:
                if name not in engine.state.metrics:
                    warnings.warn("Provided metric name '{}' is missing "
                                  "in engine's state metrics: {}".format(name, list(engine.state.metrics.keys())))
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

    def __init__(self, model, reduction=torch.norm):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Argument model should be of type torch.nn.Module, "
                            "but given {}".format(type(model)))

        if not callable(reduction):
            raise TypeError("Argument reduction should be callable, "
                            "but given {}".format(type(reduction)))

        def _is_0D_tensor(t):
            return isinstance(t, torch.Tensor) and t.ndimension() == 0

        # Test reduction function on a random tensor
        o = reduction(torch.rand(4, 2))
        if not (isinstance(o, numbers.Number) or _is_0D_tensor(o)):
            raise ValueError("Output of the reduction function should be a scalar, but got {}".format(type(o)))

        self.model = model
        self.reduction = reduction


class BaseWeightsHistHandler(BaseHandler):
    """
    Helper handler to log model's weights as histograms.
    """

    def __init__(self, model):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Argument model should be of type torch.nn.Module, "
                            "but given {}".format(type(model)))

        self.model = model
