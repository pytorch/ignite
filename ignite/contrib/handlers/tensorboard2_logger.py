try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("This contrib module requires tensorboardX to be installed.")

from typing import Callable, Dict, List, Union

from ignite.engine import Engine, Events
from torch import nn
from torch.utils.data import DataLoader


class TensorboardLogger(object):
    """
    TensorBoard handler to log metrics related to training, validation and model.

    Args:
        log_dir (str, optional): Path for logging the summary files.

    Examples:
        Create a TensorBoard summary that visualizes metrics related to training,
        validation, and model parameters, by simply attaching the handler to your engine.

        ..code-block:: python
            tbLogger = TensorboardLogger()

            tbLogger.attach(
                engine=trainer,
                name="training",
                plot_event=Events.ITERATION_COMPLETED,
                output_transform=lambda x: x,
            )
    """

    def __init__(
            self,
            log_dir=None,     # type: str,
    ):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.metrics_step = []

    def _close(self):
        """
        Closes Summary Writer
        """
        self.writer.close()

    def __del__(self):
        self._close()

    def _update(
        self,
        engine,                 # type: Engine
        name,                   # type: str
        attach_id,              # type: int
        update_period,          # type: int
        metric_names=None,      # type: Union[Dict, List]
        output_transform=None,  # type: Callable
        model=None,             # type: nn.Module
        histogram_freq=0,       # type: int
        write_grads=False,      # type: bool
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

        if not metrics:
            return

        metric_names, metric_values = list(zip(*metrics))

        for metric_name, metric_value in zip(metric_names, metric_values):
            self.writer.add_scalar(
                tag='/'.join([name, metric_name]),
                scalar_value=metric_value,
                global_step=step
            )

        if model is not None and histogram_freq > 0 and engine.state.epoch % histogram_freq == 0:
            for param_name, param in model.named_parameters():
                param_name = param_name.replace('.', '/')
                if param.requires_grad:
                    self.writer.add_histogram(
                        tag=param_name,
                        values=param.cpu().data.numpy().flatten(),
                        global_step=step
                    )

                    if write_grads:
                        self.writer.add_histogram(
                            tag=param_name + '_grad',
                            values=param.grad.cpu().data.numpy().flatten(),
                            global_step=step
                        )

    def write_graph(
            self,
            model,      # type: nn.Module
            dataloader  # type: DataLoader

    ):
        """
        Plots engine.state.metrics on epoch or iteration.
        Args:
            model (nn.Module): model to be written
            dataloader (torch.utils.DataLoader): data loader for training data
        """

        x, y = next(iter(dataloader))
        x = x.cuda() if next(model.parameters()).is_cuda else x
        self.writer.add_graph(model, x)

    def attach(
            self,
            engine,                             # type: Engine
            name,                               # type: str
            plot_event=Events.EPOCH_COMPLETED,  # type: Events
            update_period=1,                    # type: int
            metric_names=None,                  # type: Union[Dict, List]
            output_transform=None,              # type: Callable
            step_callback=None,                 # type: Callable
            model=None,                         # type: nn.Module
            histogram_freq=0,                   # type: int
            write_grads=False,                  # type: bool,
    ):
        """
        Attaches the TensorBoard event handler to an engine object.
        Args:
            engine (Engine): engine object.
            name (str): Name for the attached log.
            plot_event (str, optional): Name of event to handle.
            update_period (int, optional): Can be used to limit the number of plot updates.
            metric_names (list, optional): list of the metrics names to plot.
            output_transform (Callable, optional): a function to select what you want to plot from the engine's
                output. This function may return either a dictionary with entries in the format of ``{name: value}``,
                or a single scalar, which will be displayed with the default name `output`.
            step_callback (Callable, optional): a function to select what to use as the x value (step) from the engine's
                state. This function should return a single scalar.
            model (nn.Module, optional): model to train.
            histogram_freq (int, optional): frequency histograms should be plotted.
            write_grads (bool, optional): True, if gradients to model parameters should be visualized.
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
            name=name,
            attach_id=attach_id,
            update_period=update_period,
            metric_names=metric_names,
            output_transform=output_transform,
            model=model,
            histogram_freq=histogram_freq,
            write_grads=write_grads
        )
