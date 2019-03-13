import numbers

import warnings
import torch

from ignite.contrib.handlers.base_logger import BaseLogger, _check_output_handler_params, _setup_output_handler_metrics


__all__ = ['TensorboardLogger', 'optimizer_params_handler', 'output_handler',
           'weights_scalar_handler', 'weights_hist_handler', 'grads_scalar_handler', 'grads_hist_handler']


def optimizer_params_handler(optimizer, param_name="lr"):

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("Argument optimizer should be of type torch.optim.Optimizer, "
                        "but given {}".format(type(optimizer)))

    def wrapper(engine, logger, event_name):

        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'optimizer_params_handler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        params = {"{}/group_{}".format(param_name, i): float(param_group[param_name])
                  for i, param_group in enumerate(optimizer.param_groups)}

        for k, v in params.items():
            logger.writer.add_scalar(k, v, global_step)

    return wrapper


def output_handler(tag, metric_names=None, output_transform=None, another_engine=None):

    _check_output_handler_params(metric_names, output_transform)

    def wrapper(engine, logger, event_name):

        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'output_handler' works only with TensorboardLogger")

        metrics = _setup_output_handler_metrics(metric_names, output_transform, engine)

        state = engine.state if another_engine is None else another_engine.state
        global_step = state.get_event_attrib_value(event_name)

        for key, value in metrics.items():
            if isinstance(value, numbers.Number):
                logger.writer.add_scalar("{}/{}".format(tag, key), value, global_step)
            elif isinstance(value, torch.Tensor) and value.ndimension() == 1:
                for i, v in enumerate(value):
                    logger.writer.add_scalar("{}/{}/{}".format(tag, key, i), v.item(), global_step)
            else:
                warnings.warn("TensorboardLogger output_handler can not log "
                              "metrics value type {}".format(type(value)))

    return wrapper


def weights_scalar_handler(model, reduction=torch.norm):

    if not isinstance(model, torch.nn.Module):
        raise TypeError("Argument model should be of type torch.nn.Module, "
                        "but given {}".format(type(model)))

    if not callable(reduction):
        raise TypeError("Argument reduction should be callable, "
                        "but given {}".format(type(reduction)))

    def wrapper(engine, logger, event_name):

        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'weights_scalar_handler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        for name, p in model.named_parameters():
            name = name.replace('.', '/')
            logger.writer.add_scalar("weights_{}/{}".format(reduction.__name__, name),
                                     reduction(p.data),
                                     global_step)

    return wrapper


def weights_hist_handler(model):

    if not isinstance(model, torch.nn.Module):
        raise TypeError("Argument model should be of type torch.nn.Module, "
                        "but given {}".format(type(model)))

    def wrapper(engine, logger, event_name):

        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'weights_hist_handler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        for name, p in model.named_parameters():
            name = name.replace('.', '/')
            logger.writer.add_histogram(tag="weights/{}".format(name),
                                        values=p.data.detach().cpu().numpy(),
                                        global_step=global_step)

    return wrapper


def grads_scalar_handler(model, reduction=torch.norm):

    if not isinstance(model, torch.nn.Module):
        raise TypeError("Argument model should be of type torch.nn.Module, "
                        "but given {}".format(type(model)))

    if not callable(reduction):
        raise TypeError("Argument reduction should be callable, "
                        "but given {}".format(type(reduction)))

    def wrapper(engine, logger, event_name):

        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'grads_scalar_handler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        for name, p in model.named_parameters():
            name = name.replace('.', '/')
            logger.writer.add_scalar("grads_{}/{}".format(reduction.__name__, name),
                                     reduction(p.grad),
                                     global_step)

    return wrapper


def grads_hist_handler(model):

    if not isinstance(model, torch.nn.Module):
        raise TypeError("Argument model should be of type torch.nn.Module, "
                        "but given {}".format(type(model)))

    def wrapper(engine, logger, event_name):

        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'grads_hist_handler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        for name, p in model.named_parameters():
            name = name.replace('.', '/')
            logger.writer.add_histogram(tag="grads/{}".format(name),
                                        values=p.grad.detach().cpu().numpy(),
                                        global_step=global_step)

    return wrapper


class TensorboardLogger(BaseLogger):
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
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise RuntimeError("This contrib module requires tensorboardX to be installed.")

        self.writer = SummaryWriter(log_dir=log_dir)

    def _close(self):
        self.writer.close()
