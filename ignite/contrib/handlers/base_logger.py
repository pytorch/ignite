from abc import ABCMeta, abstractmethod

import warnings

from ignite.engine import State
from ignite._six import with_metaclass


def _check_output_handler_params(metric_names, output_transform):
    """Helper method to check input arguments of an output_handler
    """
    if metric_names is not None and not isinstance(metric_names, list):
        raise TypeError("metric_names should be a list, got {} instead.".format(type(metric_names)))

    if output_transform is not None and not callable(output_transform):
        raise TypeError("output_transform should be a function, got {} instead."
                        .format(type(output_transform)))

    if output_transform is None and metric_names is None:
        raise ValueError("Either metric_names or output_transform should be defined")


def _setup_output_handler_metrics(metric_names, output_transform, engine):
    """Helper method to setup metrics to log
    """
    metrics = {}
    if metric_names is not None:
        for name in metric_names:
            if name not in engine.state.metrics:
                warnings.warn("Provided metric name '{}' is missing "
                              "in engine's state metrics: {}".format(name, list(engine.state.metrics.keys())))
                continue
            metrics[name] = engine.state.metrics[name]

    if output_transform is not None:
        output_dict = output_transform(engine.state.output)

        if not isinstance(output_dict, dict):
            output_dict = {"output": output_dict}

        metrics.update({name: value for name, value in output_dict.items()})
    return metrics


class BaseLogger(with_metaclass(ABCMeta, object)):
    """
    Base logger handler. See implementations: TensorboardLogger, VisdomLogger, PolyaxonLogger

    """
    @abstractmethod
    def _close(self):
        pass

    def __del__(self):
        self._close()

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
