from ignite.engine import Engine
from ignite.engine import Events
import numpy as np
from typing import Callable, List

import visdom


class VisdomPlotter:
    """
    """

    def __init__(
            self,
            vis: visdom.Visdom=None,
            server: str=None,
            env: str="main",
            log_to_filename: str=None,
            save_by_default: bool=True):

        assert (vis is not None) or (server is not None), \
            "Either a visdom object or visdom server should be supplied."

        if vis is None:
            vis = visdom.Visdom(
                server=server,
                log_to_filename=log_to_filename,
            )

        if not vis.check_connection():
            raise RuntimeError("Visdom server not running. Please run python -m visdom.server")

        self.vis = vis
        self.env = env
        self.save_by_default = save_by_default
        self.plots = dict()

    def _update(
            self,
            engine: Engine,
            step: int,
            window_title: str,
            window_opts: dict,
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

        metric_names, metric_values = list(zip(*metrics))
        line_opts = window_opts.copy()
        line_opts['legend'] = metric_names

        if window_title not in self.plots:
            self.plots[window_title] = \
                self.vis.line(
                    X=np.array([step] * len(metric_values)).reshape(1, -1),
                    Y=np.array(metric_values).reshape(1, -1),
                    env=self.env,
                    opts=window_opts
                )
        else:
            self.vis.line(
                X=np.array([step] * len(metric_values)).reshape(1, -1),
                Y=np.array(metric_values).reshape(1, -1),
                env=self.env,
                opts=line_opts,
                win=self.plots[window_title],
                update='append'
            )

        if self.save_by_default:
            self.vis.save([self.env])

    def create_window(
            self,
            engine: Engine,
            window_title: str="Metrics",
            xlabel: str="epoch",
            ylabel: str="value",
            show_legend: bool=False,
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

        window_opts = dict(
            title=window_title,
            xlabel=xlabel,
            ylabel=ylabel,
            showlegend=show_legend
        )
        engine.add_event_handler(
            plot_event,
            self._update,
            engine,
            engine.state.iteration if plot_event == Events.ITERATION_COMPLETED else engine.state.epoch,
            window_title,
            window_opts,
            metric_names,
            output_transform
        )