import numbers

import warnings
import torch

from ignite.contrib.handlers.base_logger import BaseLogger, BaseOptimizerParamsHandler, BaseOutputHandler, \
    BaseWeightsScalarHandler, BaseWeightsHistHandler


__all__ = ['TensorboardLogger', 'OptimizerParamsHandler', 'OutputHandler',
           'WeightsScalarHandler', 'WeightsHistHandler', 'GradsScalarHandler', 'GradsHistHandler']


class OutputHandler(BaseOutputHandler):
    """Helper handler to log engine's output and/or metrics

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `another_engine=trainer` to take the epoch of the `trainer`
            tb_logger.attach(evaluator,
                             log_handler=OutputHandler(tag="validation",
                                                       metric_names=["nll", "accuracy"],
                                                       another_engine=trainer),
                             event_name=Events.EPOCH_COMPLETED)

    Args:
        tag (str): common title for all produced plots. For example, 'training'
        metric_names (list of str, optional): list of metric names to plot.
        output_transform (callable, optional): output transform function to prepare `engine.state.output` as a number.
            For example, `output_transform = lambda output: output`
            This function can also return a dictionary, e.g `{'loss': loss1, `another_loss`: loss2}` to label the plot
            with corresponding keys.
        another_engine (Engine): another engine to use to provide the value of event. Typically, user can provide
            the trainer if this handler is attached to an evaluator and thus it logs proper trainer's
            epoch/iteration value.
    """
    def __init__(self, tag, metric_names=None, output_transform=None, another_engine=None):
        super(OutputHandler, self).__init__(tag, metric_names, output_transform, another_engine)

    def __call__(self, engine, logger, event_name):

        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'OutputHandler' works only with TensorboardLogger")

        metrics = self._setup_output_metrics(engine)

        state = engine.state if self.another_engine is None else self.another_engine.state
        global_step = state.get_event_attrib_value(event_name)

        for key, value in metrics.items():
            if isinstance(value, numbers.Number) or \
                    isinstance(value, torch.Tensor) and value.ndimension() == 0:
                logger.writer.add_scalar("{}/{}".format(self.tag, key), value, global_step)
            elif isinstance(value, torch.Tensor) and value.ndimension() == 1:
                for i, v in enumerate(value):
                    logger.writer.add_scalar("{}/{}/{}".format(self.tag, key, i), v.item(), global_step)
            else:
                warnings.warn("TensorboardLogger output_handler can not log "
                              "metrics value type {}".format(type(value)))


class OptimizerParamsHandler(BaseOptimizerParamsHandler):
    """Helper handler to log optimizer parameters

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            tb_logger.attach(trainer,
                             log_handler=OptimizerParamsHandler(optimizer),
                             event_name=Events.ITERATION_STARTED)

    Args:
        optimizer (torch.optim.Optimizer): torch optimizer which parameters to log
        param_name (str): parameter name
    """

    def __init__(self, optimizer, param_name="lr"):
        super(OptimizerParamsHandler, self).__init__(optimizer, param_name)

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'OptimizerParamsHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        params = {"{}/group_{}".format(self.param_name, i): float(param_group[self.param_name])
                  for i, param_group in enumerate(self.optimizer.param_groups)}

        for k, v in params.items():
            logger.writer.add_scalar(k, v, global_step)


class WeightsScalarHandler(BaseWeightsScalarHandler):
    """Helper handler to log model's weights as scalars.
    Handler iterates over named parameters of the model, applies reduction function to each parameter
    produce a scalar and then logs the scalar.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the trainer to log model's weights norm after each iteration
            tb_logger.attach(trainer,
                             log_handler=WeightsScalarHandler(model, reduction=torch.norm),
                             event_name=Events.ITERATION_COMPLETED)

    Args:
        model (torch.nn.Module): model to log weights
        reduction (callable): function to reduce parameters into scalar

    """
    def __init__(self, model, reduction=torch.norm):
        super(WeightsScalarHandler, self).__init__(model, reduction)

    def __call__(self, engine, logger, event_name):

        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'WeightsScalarHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        for name, p in self.model.named_parameters():
            name = name.replace('.', '/')
            logger.writer.add_scalar("weights_{}/{}".format(self.reduction.__name__, name),
                                     self.reduction(p.data),
                                     global_step)


class WeightsHistHandler(BaseWeightsHistHandler):
    """Helper handler to log model's weights as histograms.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the trainer to log model's weights norm after each iteration
            tb_logger.attach(trainer,
                             log_handler=WeightsHistHandler(model),
                             event_name=Events.ITERATION_COMPLETED)

    Args:
        model (torch.nn.Module): model to log weights

    """

    def __init__(self, model):
        super(WeightsHistHandler, self).__init__(model)

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'WeightsHistHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        for name, p in self.model.named_parameters():
            name = name.replace('.', '/')
            logger.writer.add_histogram(tag="weights/{}".format(name),
                                        values=p.data.detach().cpu().numpy(),
                                        global_step=global_step)


class GradsScalarHandler(BaseWeightsScalarHandler):
    """Helper handler to log model's gradients as scalars.
    Handler iterates over the gradients of named parameters of the model, applies reduction function to each parameter
    produce a scalar and then logs the scalar.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the trainer to log model's weights norm after each iteration
            tb_logger.attach(trainer,
                             log_handler=GradsScalarHandler(model, reduction=torch.norm),
                             event_name=Events.ITERATION_COMPLETED)

    Args:
        model (torch.nn.Module): model to log weights
        reduction (callable): function to reduce parameters into scalar

    """
    def __init__(self, model, reduction=torch.norm):
        super(GradsScalarHandler, self).__init__(model, reduction)

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'GradsScalarHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        for name, p in self.model.named_parameters():
            name = name.replace('.', '/')
            logger.writer.add_scalar("grads_{}/{}".format(self.reduction.__name__, name),
                                     self.reduction(p.grad),
                                     global_step)


class GradsHistHandler(BaseWeightsHistHandler):
    """Helper handler to log model's gradients as histograms.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the trainer to log model's weights norm after each iteration
            tb_logger.attach(trainer,
                             log_handler=GradsHistHandler(model),
                             event_name=Events.ITERATION_COMPLETED)

    Args:
        model (torch.nn.Module): model to log weights

    """
    def __init__(self, model):
        super(GradsHistHandler, self).__init__(model)

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'GradsHistHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        for name, p in self.model.named_parameters():
            name = name.replace('.', '/')
            logger.writer.add_histogram(tag="grads/{}".format(name),
                                        values=p.grad.detach().cpu().numpy(),
                                        global_step=global_step)


class TensorboardLogger(BaseLogger):
    """
    TensorBoard handler to log metrics, model/optimizer parameters, gradients during the training and validation.

    This class requires `tensorboardX <https://github.com/lanpa/tensorboardX>`_ package to be installed:

    .. code-block:: bash

        pip install tensorboardX


    Args:
        log_dir (str): path to the directory where to log.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the trainer to log training loss at each iteration
            tb_logger.attach(trainer,
                             log_handler=OutputHandler(tag="training", output_transform=lambda loss: {'loss': loss}),
                             event_name=Events.ITERATION_COMPLETED)

            # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
            # We setup `another_engine=trainer` to take the epoch of the `trainer` instead of `train_evaluator`.
            tb_logger.attach(train_evaluator,
                             log_handler=OutputHandler(tag="training",
                                                       metric_names=["nll", "accuracy"],
                                                       another_engine=trainer),
                             event_name=Events.EPOCH_COMPLETED)

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `another_engine=trainer` to take the epoch of the `trainer` instead of `evaluator`.
            tb_logger.attach(evaluator,
                             log_handler=OutputHandler(tag="validation",
                                                       metric_names=["nll", "accuracy"],
                                                       another_engine=trainer),
                             event_name=Events.EPOCH_COMPLETED)

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            tb_logger.attach(trainer,
                             log_handler=OptimizerParamsHandler(optimizer),
                             event_name=Events.ITERATION_STARTED)

            # Attach the logger to the trainer to log model's weights norm after each iteration
            tb_logger.attach(trainer,
                             log_handler=WeightsScalarHandler(model),
                             event_name=Events.ITERATION_COMPLETED)

            # Attach the logger to the trainer to log model's weights as a histogram after each epoch
            tb_logger.attach(trainer,
                             log_handler=WeightsHistHandler(model),
                             event_name=Events.EPOCH_COMPLETED)

            # Attach the logger to the trainer to log model's gradients norm after each iteration
            tb_logger.attach(trainer,
                             log_handler=GradsScalarHandler(model),
                             event_name=Events.ITERATION_COMPLETED)

            # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
            tb_logger.attach(trainer,
                             log_handler=GradsHistHandler(model),
                             event_name=Events.EPOCH_COMPLETED)

            # We need to close the logger with we are done
            tb_logger.close()

        It is also possible to use the logger as context manager:

        .. code-block:: python

            from ignite.contrib.handlers.tensorboard_logger import *

            with TensorboardLogger(log_dir="experiments/tb_logs") as tb_logger:

                trainer = Engine(update_fn)
                # Attach the logger to the trainer to log training loss at each iteration
                tb_logger.attach(trainer,
                                 log_handler=OutputHandler(tag="training",
                                                           output_transform=lambda loss: {'loss': loss}),
                                 event_name=Events.ITERATION_COMPLETED)

    """

    def __init__(self, log_dir):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise RuntimeError("This contrib module requires tensorboardX to be installed. "
                               "Please install it with command: \n pip install tensorboardX")

        self.writer = SummaryWriter(log_dir=log_dir)

    def close(self):
        self.writer.close()
