from ignite.engine import Engine
from ignite.engine import Events
from typing import Callable, List

import mlflow


class MlflowPlotter:
    """
    """

    def _update(
            self,
            engine: Engine,
            prefix: str,
            metric_names: List=None,
            output_transform: Callable=None):

        #
        # Get all the metrics
        #
        metrics = []
        if metric_names is not None:
            if not all(metric in engine.state.metrics for metric in metric_names):
                raise KeyError("metrics not found in engine.state.metrics")

            metrics.append([(name, engine.state.metrics[name]) for name in metric_names])

        if output_transform is not None:
            output_dict = output_transform(engine.state.output)

            if not isinstance(output_dict, dict):
                output_dict = {"output": output_dict}

            metrics.append([(name, value) for name, value in output_dict.items()])

        if not metrics:
            return

        for metric_name, new_value in metrics:
            mlflow.log_metric(prefix + metric_name, new_value)

    def attach(
            self,
            engine: Engine,
            prefix: str="",
            plot_event: str=Events.EPOCH_COMPLETED,
            metric_names: List=None,
            output_transform: Callable=None):
        """
        Attaches the visdom window to an engine object
        Args:
            engine (Engine): engine object
            metric_names (list, optional): list of the metrics names to log.
            output_transform (Callable, optional): a function to select what you want to plot from the engine's
                output. This function may return either a dictionary with entries in the format of ``{name: value}``,
                or a single scalar, which will be displayed with the default name `output`.
        """
        if metric_names is not None and not isinstance(metric_names, list):
            raise TypeError("metric_names should be a list, got {} instead".format(type(metric_names)))

        if output_transform is not None and not callable(output_transform):
            raise TypeError("output_transform should be a function, got {} instead"
                            .format(type(output_transform)))

        assert plot_event in (Events.ITERATION_COMPLETED, Events.EPOCH_COMPLETED), \
            "The plotting event should be either {} or {}".format(Events.ITERATION_COMPLETED, Events.EPOCH_COMPLETED)

        engine.add_event_handler(
            plot_event,
            self._update,
            engine,
            engine.state.iteration if plot_event == Events.ITERATION_COMPLETED else engine.state.epoch,
            prefix,
            metric_names,
            output_transform
        )