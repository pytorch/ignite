from ignite.engine import Engine
from ignite.engine import Events
import numpy as np
from typing import Callable, List

import visdom


class VisdomLogger:
    """Handler that plots metrics using Visdom graphs.

    The `VisdomLogger` can be used to plot to multiple windows each one with a different
    set of metrics.

    Args:
        vis (Visdom object, optional): Visdom client.
        server (str, optinal): URL of visdom server.
        env (str, optional): Name of Visdom environment for the graphs. Defaults to "main".
        log_to_filename (str, optional): If given, the plots will be also be save to a file
            by this name. Later this graphs can be replayed from this file.
        save_by_default (bool, optional): The graphs will be saved by default by the server.

    Note:
        Either the `vis` or `server` arguments should be given to the constructor.

    Examples:

    Plotting of trainer loss.

    .. code-block:: python

        trainer = create_supervised_trainer(model, optimizer, loss)

        visdom_plotter = VisdomLogger(vis, env=env)

        visdom_plotter.create_window(
            engine=trainer,
            window_title="Training Losses",
            xlabel="epoch",
            plot_event=Events.ITERATION_COMPLETED,
            update_period=LOG_INTERVAL,
            output_transform=lambda x: {'loss": x}
        )

    Attach validation metrics

    .. code-block:: python

        metrics={
            'accuracy': CategoricalAccuracy(),
            'nll': Loss(loss)
        }
        evaluator = create_supervised_evaluator(
            model,
            metrics=metrics
        )

        visdom_plotter = VisdomLogger(vis, env=env)

        visdom_plotter.create_window(
            engine=evaluator,
            window_title="Evaluation",
            xlabel="epoch",
            plot_event=Events.EPOCH_COMPLETED,
            metric_names=list(metrics.keys())
        )

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
        self.metrics_step = []

    def _update(
            self,
            engine: Engine,
            attach_id: int,
            window_title: str,
            window_opts: dict,
            update_period: int,
            metric_names: List=None,
            output_transform: Callable=None):

        step = self.metrics_step[attach_id]
        self.metrics_step[attach_id] += 1
        if step % update_period != 0:
            return

        #
        # Get all the metrics
        #
        metrics = []
        if metric_names is not None:
            if not all(metric in engine.state.metrics for metric in metric_names):
                raise KeyError("metrics not found in engine.state.metrics")

            metrics.extend([(name, engine.state.metrics[name]) for name in metric_names])

        if output_transform is not None:
            output_dict = output_transform(engine.state.output)

            if not isinstance(output_dict, dict):
                output_dict = {"output": output_dict}

            metrics.extend([(name, value) for name, value in output_dict.items()])

        if not metrics:
            return

        metric_names, metric_values = list(zip(*metrics))
        line_opts = window_opts.copy()
        line_opts['legend'] = list(metric_names)

        if window_title not in self.plots:
            win = self.vis.line(
                    X=np.array([step] * len(metric_values)).reshape(1, -1),
                    Y=np.array(metric_values).reshape(1, -1),
                    env=self.env,
                    opts=line_opts
                )
            self.plots[window_title] = win

        else:
            win = self.plots[window_title]
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
            update_period: int=1,
            metric_names: List=None,
            output_transform: Callable=None):
        """
        Creates a Visdom window and attaches it to an engine object

        Args:
            engine (Engine): engine object
            window_title (str, optional): The title that will given to the window.
            xlabel (str, optional): Label of the x-axis.
            ylabel (str, optional): Label of the y-axis.
            show_legend (bool, optional): Whether to add a legend to the window,
            plot_event (str, optional): Name of event to handle.
            update_period (int, optional): Can be used to limit the number of plot updates.
            metric_names (list, optional): list of the metrics names to plot.
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

        attach_id = len(self.metrics_step)
        self.metrics_step.append(0)

        engine.add_event_handler(
            plot_event,
            self._update,
            attach_id=attach_id,
            window_title=window_title,
            window_opts=window_opts,
            update_period=update_period,
            metric_names=metric_names,
            output_transform=output_transform
        )