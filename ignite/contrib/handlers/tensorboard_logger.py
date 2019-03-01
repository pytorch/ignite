import torch

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("This contrib module requires tensorboardX to be installed.")

from ignite.engine import Events


__all__ = ['TensorboardLogger', 'optimizer_params_handler', 'output_handler',
           'weights_scalar_handler', 'weights_hist_handler', 'grads_scalar_handler', 'grads_hist_handler']


# TODO: Move the mapping to State
MAP_EVENT_TO_STATE_ATTR = {
    Events.ITERATION_STARTED: "iteration",
    Events.ITERATION_COMPLETED: "iteration",
    Events.EPOCH_STARTED: "epoch",
    Events.EPOCH_COMPLETED: "epoch",
    Events.STARTED: "epoch",
    Events.COMPLETED: "epoch"
}


def optimizer_params_handler(optimizer, param_name="lr"):

    def wrapper(engine, writer, state_attr):
        global_step = getattr(engine.state, state_attr)
        params = {"{}/group_{}".format(param_name, i): float(param_group[param_name])
                  for i, param_group in enumerate(optimizer.param_groups)}

        for k, v in params.items():
            writer.add_scalar(k, v, global_step)

    return wrapper


def output_handler(tag, metric_names=None, output_transform=None, another_engine=None):
    if metric_names is not None and not isinstance(metric_names, list):
        raise TypeError("metric_names should be a list, got {} instead.".format(type(metric_names)))

    if output_transform is not None and not callable(output_transform):
        raise TypeError("output_transform should be a function, got {} instead."
                        .format(type(output_transform)))

    if output_transform is None and metric_names is None:
        raise ValueError("Either metric_names or output_transform should be defined")

    def wrapper(engine, writer, state_attr):

        metrics = {}
        if metric_names is not None:
            if not all(metric in engine.state.metrics for metric in metric_names):
                # -> Maybe display a warning ?
                pass
            else:
                metrics.update({name: engine.state.metrics[name] for name in metric_names})

        if output_transform is not None:
            output_dict = output_transform(engine.state.output)

            if not isinstance(output_dict, dict):
                output_dict = {"output": output_dict}

            metrics.update({name: value for name, value in output_dict.items()})

        state = engine.state if another_engine is None else another_engine.state
        global_step = getattr(state, state_attr)

        for k, v in metrics.items():
            writer.add_scalar("{}/{}".format(tag, k), v, global_step)

    return wrapper


def weights_scalar_handler(model, reduction=torch.norm):
    def wrapper(engine, writer, state_attr):
        global_step = getattr(engine.state, state_attr)
        for name, p in model.named_parameters():
            name = name.replace('.', '/')
            writer.add_scalar("weights_{}/{}".format(reduction.__name__, name),
                              reduction(p.data),
                              global_step)

    return wrapper


def weights_hist_handler(model):
    def wrapper(engine, writer, state_attr):
        global_step = getattr(engine.state, state_attr)
        for name, p in model.named_parameters():
            name = name.replace('.', '/')
            writer.add_histogram(tag="weights/{}".format(name),
                                 values=p.data.detach().cpu().numpy(),
                                 global_step=global_step)

    return wrapper


def grads_scalar_handler(model, reduction=torch.norm):
    def wrapper(engine, writer, state_attr):
        global_step = getattr(engine.state, state_attr)
        for name, p in model.named_parameters():
            name = name.replace('.', '/')
            writer.add_scalar("grads_{}/{}".format(reduction.__name__, name),
                              reduction(p.grad),
                              global_step)

    return wrapper


def grads_hist_handler(model):
    def wrapper(engine, writer, state_attr):
        global_step = getattr(engine.state, state_attr)
        for name, p in model.named_parameters():
            name = name.replace('.', '/')
            writer.add_histogram(tag="grads/{}".format(name),
                                 values=p.grad.detach().cpu().numpy(),
                                 global_step=global_step)

    return wrapper


class TensorboardLogger(object):
    """
    TensorBoard handler to log metrics, model/optimizer parameters, gradients during the training and validation.

    Args:
        log_dir (str): path to the directory where to log.

    Examples:

        ..code-block:: python

            from ignite.contrib.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the trainer to log training loss at each iteration
            tb_logger.attach(trainer,
                             log_handler=output_handler(tag="training", output_transform=lambda loss: {'loss': loss}),
                             event_name=Events.ITERATION_COMPLETED)

            # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
            # We setup `another_engine=trainer` to take the epoch of the `trainer` instead of `train_evaluator`.
            tb_logger.attach(train_evaluator,
                             log_handler=output_handler(tag="training",
                                                        metric_names=["nll", "accuracy"],
                                                        another_engine=trainer),
                             event_name=Events.EPOCH_COMPLETED)

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `another_engine=trainer` to take the epoch of the `trainer` instead of `evaluator`.
            tb_logger.attach(evaluator,
                             log_handler=output_handler(tag="validation",
                                                        metric_names=["nll", "accuracy"],
                                                        another_engine=trainer),
                             event_name=Events.EPOCH_COMPLETED)

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            tb_logger.attach(trainer,
                             log_handler=optimizer_params_handler(optimizer),
                             event_name=Events.ITERATION_COMPLETED)

            # Log model's graph
            x, _ = next(iter(train_loader))
            tb_logger.log_graph(model, x.to(device))

            # Attach the logger to the trainer to log model's weights norm after each iteration
            tb_logger.attach(trainer,
                             log_handler=weights_scalar_handler(model),
                             event_name=Events.ITERATION_COMPLETED)

            # Attach the logger to the trainer to log model's weights as a histogram after each epoch
            tb_logger.attach(trainer,
                             log_handler=weights_hist_handler(model),
                             event_name=Events.EPOCH_COMPLETED)

            # Attach the logger to the trainer to log model's gradients norm after each iteration
            tb_logger.attach(trainer,
                             log_handler=grads_scalar_handler(model),
                             event_name=Events.ITERATION_COMPLETED)

            # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
            tb_logger.attach(trainer,
                             log_handler=grads_hist_handler(model),
                             event_name=Events.EPOCH_COMPLETED)

    """

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def _close(self):
        self.writer.close()

    def __del__(self):
        self._close()

    def log_graph(self, model, input_to_model, **kwargs):
        """Log model's graph

        Args:
            model (torch.nn.Module): input model
            input_to_model (torch.Tensor or torch.utils.data.DataLoader): input to the model
            **kwargs: kwargs to `tensorboardX.SummaryWriter.add_graph`

        """
        from torch.utils.data import DataLoader

        if isinstance(input_to_model, DataLoader):
            input_to_model, _ = next(iter(input_to_model))
        self.writer.add_graph(model, input_to_model, **kwargs)

    def attach(self, engine, log_handler, event_name):
        """Attach the logger to the engine and execute `log_handler` function at `event_name` events.

        Args:
            engine (Engine): engine object.
            log_handler (callable): a logging handler to execute
            event_name: event to attach the logging handler to. Valid events are from :class:`~ignite.engine.Events`
                or any `event_name` added by :meth:`~ignite.engine.Engine.register_events`.

        """
        engine.add_event_handler(event_name, log_handler, self.writer, MAP_EVENT_TO_STATE_ATTR[event_name])
