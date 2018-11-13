from ignite.engine import Engine
from ignite.engine import Events
import os
import numpy as np
from typing import Callable, Dict, List, Union


class _VisdomWindows:
    """Object encapsulating a Visdom Window.

    Args:
        vis (Visdom object): Visdom client.
        env (str): Name of Visdom environment for the graphs.
        save_by_default (bool): The graphs will be saved by default by the server.
        window_title (str, optional): The title that will given to the window.
        xlabel (str, optional): Label of the x-axis.
        ylabel (str, optional): Label of the y-axis.
        show_legend (bool, optional): Whether to add a legend to the window,

    Note:
        Should not be instantiated directly by the user.
    """

    def __init__(
        self,
        vis,                     # type: visdom.Visdom
        env,                     # type: str
        save_by_default,         # type: bool
        window_title="Metrics",  # type: str
        xlabel="Epoch",          # type: str
        ylabel="Value",          # type: str
        show_legend=False,       # type: bool
    ):

        self.window_opts = dict(
            title=window_title,
            xlabel=xlabel,
            ylabel=ylabel,
            showlegend=show_legend
        )

        self.vis = vis
        self.env = env
        self.save_by_default = save_by_default
        self.win = None
        self.metrics_step = []

    def _update(
        self,
        engine,                 # type: Engine
        attach_id,              # type: int
        update_period,          # type: int
        metric_names=None,      # type: Union[Dict, List]
        output_transform=None,  # type: Callable
        param_history=False     # type: bool
    ):

        step = self.metrics_step[attach_id]
        if type(step) is int:
            self.metrics_step[attach_id] += 1
            if step % update_period != 0:
                return
        else:
            step = step(engine.state)
        #
        # Get all the metrics
        #
        metrics = []
        if metric_names is not None:
            if isinstance(metric_names, dict):
                metric_names = metric_names.items()
            else:
                metric_names = [(n, n) for n in metric_names]

            if not all(name in engine.state.metrics for _, name in metric_names):
                raise KeyError("metrics not found in engine.state.metrics")

            metrics.extend(
                [(label, engine.state.metrics[name]) for label, name in metric_names]
            )

        if output_transform is not None:
            output_dict = output_transform(engine.state.output)

            if not isinstance(output_dict, dict):
                output_dict = {"output": output_dict}

            metrics.extend([(name, value) for name, value in output_dict.items()])

        if param_history:
            metrics.extend([(name, value[-1][0]) for name, value in engine.state.param_history.items()])

        if not metrics:
            return

        metric_names, metric_values = list(zip(*metrics))
        line_opts = self.window_opts.copy()

        if self.win is None:
            update = None
        else:
            update = 'append'

        for metric_name, metric_value in zip(metric_names, metric_values):
            self.win = self.vis.line(
                X=np.array([step]),
                Y=np.array([metric_value]),
                env=self.env,
                win=self.win,
                update=update,
                opts=line_opts,
                name=metric_name
            )

        if self.save_by_default:
            self.vis.save([self.env])

    def attach(
        self,
        engine,                             # type: Engine
        plot_event=Events.EPOCH_COMPLETED,  # type: Events
        update_period=1,                    # type: int
        metric_names=None,                  # type: Union[Dict, List]
        output_transform=None,              # type: Callable
        param_history=False,                # type: bool
        step_callback=None,                 # type: Callable
    ):
        """
        Creates a Visdom window and attaches it to an engine object

        Args:
            engine (Engine): engine object
            plot_event (str, optional): Name of event to handle.
            update_period (int, optional): Can be used to limit the number of plot updates.
            metric_names (list, optional): list of the metrics names to plot.
            output_transform (Callable, optional): a function to select what you want to plot from the engine's
                output. This function may return either a dictionary with entries in the format of ``{name: value}``,
                or a single scalar, which will be displayed with the default name `output`.
            param_history (bool, optional): If true, will plot all the parameters logged in `param_history`.
            step_callback (Callable, optional): a function to select what to use as the x value (step) from the engine's
                state. This function should return a single scalar.
        """
        if metric_names is not None and \
                not (isinstance(metric_names, list) or isinstance(metric_names, dict)):
            raise TypeError("metric_names should be a list or dict, "
                            "got {} instead".format(type(metric_names)))

        if output_transform is not None and not callable(output_transform):
            raise TypeError("output_transform should be a function, got {} instead"
                            .format(type(output_transform)))

        if step_callback is not None and not callable(step_callback):
            raise TypeError("step_callback should be a function, got {} instead"
                            .format(type(step_callback)))

        assert plot_event in (Events.ITERATION_COMPLETED, Events.EPOCH_COMPLETED), \
            "The plotting event should be either {} or {}".format(Events.ITERATION_COMPLETED, Events.EPOCH_COMPLETED)

        attach_id = len(self.metrics_step)

        if step_callback is None:
            self.metrics_step.append(0)
        else:
            self.metrics_step.append(step_callback)

        engine.add_event_handler(
            plot_event,
            self._update,
            attach_id=attach_id,
            update_period=update_period,
            metric_names=metric_names,
            output_transform=output_transform,
            param_history=param_history
        )


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
        The visdom server can be set by passing an already configured visdom client (using
        the `vis` argument). Alternatively, the URL of the server can be passed using
        the `server` argument or by setting the `VISDOM_SERVER_URL` environment variable.
        By default, when none of these methods is used, the constructor will try to connect
        to `http://localhost`.

    Examples:

    Plotting of trainer loss.

    .. code-block:: python

        trainer = create_supervised_trainer(model, optimizer, loss)

        visdom_plotter = VisdomLogger()

        visdom_plotter.create_window(
            window_title="Training Losses",
            xlabel="epoch",
        ).attach(
            engine=trainer,
            plot_event=Events.ITERATION_COMPLETED,
            update_period=LOG_INTERVAL,
            output_transform=lambda x: {"loss": x}
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
        ).attach(
            window_title="Evaluation",
            metric_names=list(metrics.keys())
        )

    """

    def __init__(
        self,
        vis=None,              # type: visdom.Visdom
        server=None,           # type: str
        env="main",            # type: str
        log_to_filename=None,  # type: str
        save_by_default=True,  # type: bool
    ):

        try:
            import visdom
        except ImportError:
            raise RuntimeError("No visdom package is found. Please install it with command: \n pip install visdom")

        if vis is None:
            if server is None:
                server = os.environ.get("VISDOM_SERVER_URL", 'http://localhost')

            vis = visdom.Visdom(
                server=server,
                log_to_filename=log_to_filename,
            )

        if not vis.check_connection():
            raise RuntimeError("Failed to connect to Visdom server at {}. "
                               "Did you run python -m visdom.server ?".format(server))

        self.vis = vis
        self.env = env
        self.save_by_default = save_by_default

    def create_window(
        self,
        window_title="Metrics",             # type: str
        xlabel="Epoch",                     # type: str
        ylabel="Value",                     # type: str
        show_legend=False,                  # type: bool
    ):
        """
        Creates a Visdom window and attaches it to an engine object

        Args:
            window_title (str, optional): The title that will given to the window.
            xlabel (str, optional): Label of the x-axis.
            ylabel (str, optional): Label of the y-axis.
            show_legend (bool, optional): Whether to add a legend to the window,
        """

        return _VisdomWindows(
            vis=self.vis,
            env=self.env,
            save_by_default=self.save_by_default,
            window_title=window_title,
            xlabel=xlabel,
            ylabel=ylabel,
            show_legend=show_legend
        )
