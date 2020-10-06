import numbers
import warnings
from typing import Any, Callable, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ignite.contrib.handlers.base_logger import (
    BaseLogger,
    BaseOptimizerParamsHandler,
    BaseOutputHandler,
    BaseWeightsHistHandler,
    BaseWeightsScalarHandler,
)
from ignite.engine import Engine, EventEnum
from ignite.handlers import global_step_from_engine

__all__ = [
    "TensorboardLogger",
    "OptimizerParamsHandler",
    "OutputHandler",
    "WeightsScalarHandler",
    "WeightsHistHandler",
    "GradsScalarHandler",
    "GradsHistHandler",
    "global_step_from_engine",
]


class TensorboardLogger(BaseLogger):
    """
    TensorBoard handler to log metrics, model/optimizer parameters, gradients during the training and validation.

    By default, this class favors `tensorboardX <https://github.com/lanpa/tensorboardX>`_ package if installed:

    .. code-block:: bash

        pip install tensorboardX

    otherwise, it falls back to using
    `PyTorch's SummaryWriter
    <https://pytorch.org/docs/stable/tensorboard.html>`_
    (>=v1.2.0).

    Args:
        *args: Positional arguments accepted from
            `SummaryWriter
            <https://pytorch.org/docs/stable/tensorboard.html>`_.
        **kwargs: Keyword arguments accepted from
            `SummaryWriter
            <https://pytorch.org/docs/stable/tensorboard.html>`_.
            For example, `log_dir` to setup path to the directory where to log.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the trainer to log training loss at each iteration
            tb_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="training",
                output_transform=lambda loss: {"loss": loss}
            )

            # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
            # We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer` instead of `train_evaluator`.
            tb_logger.attach_output_handler(
                train_evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="training",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer),
            )

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
            # `trainer` instead of `evaluator`.
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer)),
            )

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            tb_logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                optimizer=optimizer,
                param_name='lr'  # optional
            )

            # Attach the logger to the trainer to log model's weights norm after each iteration
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(model)
            )

            # Attach the logger to the trainer to log model's weights as a histogram after each epoch
            tb_logger.attach(
                trainer,
                event_name=Events.EPOCH_COMPLETED,
                log_handler=WeightsHistHandler(model)
            )

            # Attach the logger to the trainer to log model's gradients norm after each iteration
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsScalarHandler(model)
            )

            # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
            tb_logger.attach(
                trainer,
                event_name=Events.EPOCH_COMPLETED,
                log_handler=GradsHistHandler(model)
            )

            # We need to close the logger with we are done
            tb_logger.close()

        It is also possible to use the logger as context manager:

        .. code-block:: python

            from ignite.contrib.handlers.tensorboard_logger import *

            with TensorboardLogger(log_dir="experiments/tb_logs") as tb_logger:

                trainer = Engine(update_fn)
                # Attach the logger to the trainer to log training loss at each iteration
                tb_logger.attach_output_handler(
                    trainer,
                    event_name=Events.ITERATION_COMPLETED,
                    tag="training",
                    output_transform=lambda loss: {"loss": loss}
                )

    """

    def __init__(self, *args: Any, **kwargs: Any):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise RuntimeError(
                    "This contrib module requires either tensorboardX or torch >= 1.2.0. "
                    "You may install tensorboardX with command: \n pip install tensorboardX \n"
                    "or upgrade PyTorch using your package manager of choice (pip or conda)."
                )

        self.writer = SummaryWriter(*args, **kwargs)

    def close(self):
        self.writer.close()

    def _create_output_handler(self, *args: Any, **kwargs: Any):
        return OutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args: Any, **kwargs: Any):
        return OptimizerParamsHandler(*args, **kwargs)


class OutputHandler(BaseOutputHandler):
    """Helper handler to log engine's output and/or metrics

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer`:
            tb_logger.attach(
                evaluator,
                log_handler=OutputHandler(
                    tag="validation",
                    metric_names=["nll", "accuracy"],
                    global_step_transform=global_step_from_engine(trainer)
                ),
                event_name=Events.EPOCH_COMPLETED
            )
            # or equivalently
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer)
            )

        Another example, where model is evaluated every 500 iterations:

        .. code-block:: python

            from ignite.contrib.handlers.tensorboard_logger import *

            @trainer.on(Events.ITERATION_COMPLETED(every=500))
            def evaluate(engine):
                evaluator.run(validation_set, max_epochs=1)

            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            def global_step_transform(*args, **kwargs):
                return trainer.state.iteration

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # every 500 iterations. Since evaluator engine does not have access to the training iteration, we
            # provide a global_step_transform to return the trainer.state.iteration for the global_step, each time
            # evaluator metrics are plotted on Tensorboard.

            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metrics=["nll", "accuracy"],
                global_step_transform=global_step_transform
            )

    Args:
        tag (str): common title for all produced plots. For example, "training"
        metric_names (list of str, optional): list of metric names to plot or a string "all" to plot all available
            metrics.
        output_transform (callable, optional): output transform function to prepare `engine.state.output` as a number.
            For example, `output_transform = lambda output: output`
            This function can also return a dictionary, e.g `{"loss": loss1, "another_loss": loss2}` to label the plot
            with corresponding keys.
        global_step_transform (callable, optional): global step transform function to output a desired global step.
            Input of the function is `(engine, event_name)`. Output of function should be an integer.
            Default is None, global_step based on attached engine. If provided,
            uses function output as global_step. To setup global step from another engine, please use
            :meth:`~ignite.contrib.handlers.tensorboard_logger.global_step_from_engine`.

    Note:

        Example of `global_step_transform`:

        .. code-block:: python

            def global_step_transform(engine, event_name):
                return engine.state.get_event_attrib_value(event_name)

    """

    def __init__(
        self,
        tag: str,
        metric_names: Optional[List[str]] = None,
        output_transform: Optional[Callable] = None,
        global_step_transform: Optional[Callable] = None,
    ):
        super(OutputHandler, self).__init__(tag, metric_names, output_transform, global_step_transform)

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, EventEnum]):

        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'OutputHandler' works only with TensorboardLogger")

        metrics = self._setup_output_metrics(engine)

        global_step = self.global_step_transform(engine, event_name)
        if not isinstance(global_step, int):
            raise TypeError(
                "global_step must be int, got {}."
                " Please check the output of global_step_transform.".format(type(global_step))
            )

        for key, value in metrics.items():
            if isinstance(value, numbers.Number) or isinstance(value, torch.Tensor) and value.ndimension() == 0:
                logger.writer.add_scalar("{}/{}".format(self.tag, key), value, global_step)
            elif isinstance(value, torch.Tensor) and value.ndimension() == 1:
                for i, v in enumerate(value):
                    logger.writer.add_scalar("{}/{}/{}".format(self.tag, key, i), v.item(), global_step)
            else:
                warnings.warn("TensorboardLogger output_handler can not log metrics value type {}".format(type(value)))


class OptimizerParamsHandler(BaseOptimizerParamsHandler):
    """Helper handler to log optimizer parameters

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            tb_logger.attach(
                trainer,
                log_handler=OptimizerParamsHandler(optimizer),
                event_name=Events.ITERATION_STARTED
            )
            # or equivalently
            tb_logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                optimizer=optimizer
            )

    Args:
        optimizer (torch.optim.Optimizer or object): torch optimizer or any object with attribute ``param_groups``
            as a sequence.
        param_name (str): parameter name
        tag (str, optional): common title for all produced plots. For example, "generator"
    """

    def __init__(self, optimizer: Optimizer, param_name: str = "lr", tag: Optional[str] = None):
        super(OptimizerParamsHandler, self).__init__(optimizer, param_name, tag)

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, EventEnum]):
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler OptimizerParamsHandler works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        params = {
            "{}{}/group_{}".format(tag_prefix, self.param_name, i): float(param_group[self.param_name])
            for i, param_group in enumerate(self.optimizer.param_groups)
        }

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
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(model, reduction=torch.norm)
            )

    Args:
        model (torch.nn.Module): model to log weights
        reduction (callable): function to reduce parameters into scalar
        tag (str, optional): common title for all produced plots. For example, "generator"

    """

    def __init__(self, model: nn.Module, reduction: Callable = torch.norm, tag: Optional[str] = None):
        super(WeightsScalarHandler, self).__init__(model, reduction, tag=tag)

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, EventEnum]):

        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'WeightsScalarHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue

            name = name.replace(".", "/")
            logger.writer.add_scalar(
                "{}weights_{}/{}".format(tag_prefix, self.reduction.__name__, name), self.reduction(p.data), global_step
            )


class WeightsHistHandler(BaseWeightsHistHandler):
    """Helper handler to log model's weights as histograms.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the trainer to log model's weights norm after each iteration
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsHistHandler(model)
            )

    Args:
        model (torch.nn.Module): model to log weights
        tag (str, optional): common title for all produced plots. For example, "generator"

    """

    def __init__(self, model: nn.Module, tag: Optional[str] = None):
        super(WeightsHistHandler, self).__init__(model, tag=tag)

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, EventEnum]):
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'WeightsHistHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue

            name = name.replace(".", "/")
            logger.writer.add_histogram(
                tag="{}weights/{}".format(tag_prefix, name),
                values=p.data.detach().cpu().numpy(),
                global_step=global_step,
            )


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
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsScalarHandler(model, reduction=torch.norm)
            )

    Args:
        model (torch.nn.Module): model to log weights
        reduction (callable): function to reduce parameters into scalar
        tag (str, optional): common title for all produced plots. For example, "generator"

    """

    def __init__(self, model: nn.Module, reduction: Callable = torch.norm, tag: Optional[str] = None):
        super(GradsScalarHandler, self).__init__(model, reduction, tag=tag)

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, EventEnum]):
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'GradsScalarHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue

            name = name.replace(".", "/")
            logger.writer.add_scalar(
                "{}grads_{}/{}".format(tag_prefix, self.reduction.__name__, name), self.reduction(p.grad), global_step
            )


class GradsHistHandler(BaseWeightsHistHandler):
    """Helper handler to log model's gradients as histograms.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the trainer to log model's weights norm after each iteration
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsHistHandler(model)
            )

    Args:
        model (torch.nn.Module): model to log weights
        tag (str, optional): common title for all produced plots. For example, "generator"

    """

    def __init__(self, model: nn.Module, tag: Optional[str] = None):
        super(GradsHistHandler, self).__init__(model, tag=tag)

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, EventEnum]):
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'GradsHistHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue

            name = name.replace(".", "/")
            logger.writer.add_histogram(
                tag="{}grads/{}".format(tag_prefix, name), values=p.grad.detach().cpu().numpy(), global_step=global_step
            )
