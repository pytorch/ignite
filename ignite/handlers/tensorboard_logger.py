"""TensorBoard logger and its helper handlers."""

from typing import Any, Callable, List, Optional, Union

from torch.optim import Optimizer

from ignite.engine import Engine, Events

from ignite.handlers.base_logger import (
    BaseLogger,
    BaseOptimizerParamsHandler,
    BaseOutputHandler,
    BaseWeightsHandler,
    BaseWeightsScalarHandler,
)
from ignite.handlers.utils import global_step_from_engine  # noqa

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
        args: Positional arguments accepted from
            `SummaryWriter
            <https://pytorch.org/docs/stable/tensorboard.html>`_.
        kwargs: Keyword arguments accepted from
            `SummaryWriter
            <https://pytorch.org/docs/stable/tensorboard.html>`_.
            For example, `log_dir` to setup path to the directory where to log.

    Examples:
        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

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

            # We need to close the logger when we are done
            tb_logger.close()

        It is also possible to use the logger as context manager:

        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

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
                raise ModuleNotFoundError(
                    "This contrib module requires either tensorboardX or torch >= 1.2.0. "
                    "You may install tensorboardX with command: \n pip install tensorboardX \n"
                    "or upgrade PyTorch using your package manager of choice (pip or conda)."
                )

        self.writer = SummaryWriter(*args, **kwargs)

    def __getattr__(self, attr: Any) -> Any:
        return getattr(self.writer, attr)

    def close(self) -> None:
        self.writer.close()

    def _create_output_handler(self, *args: Any, **kwargs: Any) -> "OutputHandler":
        return OutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args: Any, **kwargs: Any) -> "OptimizerParamsHandler":
        return OptimizerParamsHandler(*args, **kwargs)


class OutputHandler(BaseOutputHandler):
    """Helper handler to log engine's output, engine's state attributes and/or metrics

    Args:
        tag: common title for all produced plots. For example, "training"
        metric_names: list of metric names to plot or a string "all" to plot all available
            metrics.
        output_transform: output transform function to prepare `engine.state.output` as a number.
            For example, `output_transform = lambda output: output`
            This function can also return a dictionary, e.g `{"loss": loss1, "another_loss": loss2}` to label the plot
            with corresponding keys.
        global_step_transform: global step transform function to output a desired global step.
            Input of the function is `(engine, event_name)`. Output of function should be an integer.
            Default is None, global_step based on attached engine. If provided,
            uses function output as global_step. To setup global step from another engine, please use
            :meth:`~ignite.handlers.tensorboard_logger.global_step_from_engine`.
        state_attributes: list of attributes of the ``trainer.state`` to plot.

    Examples:
        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

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

            from ignite.handlers.tensorboard_logger import *

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

        Another example where the State Attributes ``trainer.state.alpha`` and ``trainer.state.beta``
        are also logged along with the NLL and Accuracy after each iteration:

        .. code-block:: python

            tb_logger.attach(
                trainer,
                log_handler=OutputHandler(
                    tag="training",
                    metric_names=["nll", "accuracy"],
                    state_attributes=["alpha", "beta"],
                ),
                event_name=Events.ITERATION_COMPLETED
            )

        Example of `global_step_transform`:

        .. code-block:: python

            def global_step_transform(engine, event_name):
                return engine.state.get_event_attrib_value(event_name)

    .. versionchanged:: 0.4.7
        accepts an optional list of `state_attributes`
    """

    def __init__(
        self,
        tag: str,
        metric_names: Optional[List[str]] = None,
        output_transform: Optional[Callable] = None,
        global_step_transform: Optional[Callable[[Engine, Union[str, Events]], int]] = None,
        state_attributes: Optional[List[str]] = None,
    ):
        super(OutputHandler, self).__init__(
            tag, metric_names, output_transform, global_step_transform, state_attributes
        )

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'OutputHandler' works only with TensorboardLogger")

        metrics = self._setup_output_metrics_state_attrs(engine, key_tuple=False)

        global_step = self.global_step_transform(engine, event_name)
        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}."
                " Please check the output of global_step_transform."
            )

        for key, value in metrics.items():
            logger.writer.add_scalar(key, value, global_step)


class OptimizerParamsHandler(BaseOptimizerParamsHandler):
    """Helper handler to log optimizer parameters

    Args:
        optimizer: torch optimizer or any object with attribute ``param_groups``
            as a sequence.
        param_name: parameter name
        tag: common title for all produced plots. For example, "generator"

    Examples:
        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

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
    """

    def __init__(self, optimizer: Optimizer, param_name: str = "lr", tag: Optional[str] = None):
        super(OptimizerParamsHandler, self).__init__(optimizer, param_name, tag)

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler OptimizerParamsHandler works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        params = {
            f"{tag_prefix}{self.param_name}/group_{i}": float(param_group[self.param_name])
            for i, param_group in enumerate(self.optimizer.param_groups)
        }

        for k, v in params.items():
            logger.writer.add_scalar(k, v, global_step)


class WeightsScalarHandler(BaseWeightsScalarHandler):
    """Helper handler to log model's weights as scalars.
    Handler, upon construction, iterates over named parameters of the model and keep
    reference to ones permitted by `whitelist`. Then at every call, applies
    reduction function to each parameter, produces a scalar and logs it.

    Args:
        model: model to log weights
        reduction: function to reduce parameters into scalar
        tag: common title for all produced plots. For example, "generator"
        whitelist: specific weights to log. Should be list of model's submodules
            or parameters names, or a callable which gets weight along with its name
            and determines if it should be logged. Names should be fully-qualified.
            For more information please refer to `PyTorch docs
            <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_submodule>`_.
            If not given, all of model's weights are logged.

    Examples:
        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the trainer to log model's weights norm after each iteration
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(model, reduction=torch.norm)
            )

        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Log only `fc` weights
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(
                    model,
                    whitelist=['fc']
                )
            )

        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Log weights which have `bias` in their names
            def has_bias_in_name(n, p):
                return 'bias' in n

            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(model, whitelist=has_bias_in_name)
            )

    ..  versionchanged:: 0.4.9
        optional argument `whitelist` added.
    """

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'WeightsScalarHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        for name, p in self.weights:
            name = name.replace(".", "/")
            logger.writer.add_scalar(
                f"{tag_prefix}weights_{self.reduction.__name__}/{name}",
                self.reduction(p.data),
                global_step,
            )


class WeightsHistHandler(BaseWeightsHandler):
    """Helper handler to log model's weights as histograms.

    Args:
        model: model to log weights
        tag: common title for all produced plots. For example, "generator"
        whitelist: specific weights to log. Should be list of model's submodules
            or parameters names, or a callable which gets weight along with its name
            and determines if it should be logged. Names should be fully-qualified.
            For more information please refer to `PyTorch docs
            <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_submodule>`_.
            If not given, all of model's weights are logged.

    Examples:
        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the trainer to log model's weights norm after each iteration
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsHistHandler(model)
            )

        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Log weights of `fc` layer
            weights = ['fc']

            # Attach the logger to the trainer to log weights norm after each iteration
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsHistHandler(model, whitelist=weights)
            )

        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Log weights which name include 'conv'.
            weight_selector = lambda name, p: 'conv' in name

            # Attach the logger to the trainer to log weights norm after each iteration
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsHistHandler(model, whitelist=weight_selector)
            )

    ..  versionchanged:: 0.4.9
        optional argument `whitelist` added.
    """

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'WeightsHistHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        for name, p in self.weights:
            name = name.replace(".", "/")
            logger.writer.add_histogram(
                tag=f"{tag_prefix}weights/{name}", values=p.data.cpu().numpy(), global_step=global_step
            )


class GradsScalarHandler(BaseWeightsScalarHandler):
    """Helper handler to log model's gradients as scalars.
    Handler, upon construction, iterates over named parameters of the model and keep
    reference to ones permitted by the `whitelist`. Then at every call, applies
    reduction function to each parameter's gradient, produces a scalar and logs it.

    Args:
        model: model to log weights
        reduction: function to reduce parameters into scalar
        tag: common title for all produced plots. For example, "generator"
        whitelist: specific gradients to log. Should be list of model's submodules
            or parameters names, or a callable which gets weight along with its name
            and determines if its gradient should be logged. Names should be
            fully-qualified. For more information please refer to `PyTorch docs
            <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_submodule>`_.
            If not given, all of model's gradients are logged.

    Examples:
        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the trainer to log model's gradients norm after each iteration
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsScalarHandler(model, reduction=torch.norm)
            )

        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Log gradient of `base`
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsScalarHandler(
                    model,
                    reduction=torch.norm,
                    whitelist=['base']
                )
            )

        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Log gradient of weights which belong to a `fc` layer
            def is_in_fc_layer(n, p):
                return 'fc' in n

            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsScalarHandler(model, whitelist=is_in_fc_layer)
            )

    ..  versionchanged:: 0.4.9
        optional argument `whitelist` added.
    """

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'GradsScalarHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        for name, p in self.weights:
            if p.grad is None:
                continue

            name = name.replace(".", "/")
            logger.writer.add_scalar(
                f"{tag_prefix}grads_{self.reduction.__name__}/{name}", self.reduction(p.grad), global_step
            )


class GradsHistHandler(BaseWeightsHandler):
    """Helper handler to log model's gradients as histograms.

    Args:
        model: model to log weights
        tag: common title for all produced plots. For example, "generator"
        whitelist: specific gradients to log. Should be list of model's submodules
            or parameters names, or a callable which gets weight along with its name
            and determines if its gradient should be logged. Names should be
            fully-qualified. For more information please refer to `PyTorch docs
            <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_submodule>`_.
            If not given, all of model's gradients are logged.

    Examples:
        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            # Create a logger
            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Attach the logger to the trainer to log model's weights norm after each iteration
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsHistHandler(model)
            )

        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Log gradient of `fc.bias`
            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsHistHandler(model, whitelist=['fc.bias'])
            )

        .. code-block:: python

            from ignite.handlers.tensorboard_logger import *

            tb_logger = TensorboardLogger(log_dir="experiments/tb_logs")

            # Log gradient of weights which have shape (2, 1)
            def has_shape_2_1(n, p):
                return p.shape == (2,1)

            tb_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsHistHandler(model, whitelist=has_shape_2_1)
            )

    ..  versionchanged:: 0.4.9
        optional argument `whitelist` added.
    """

    def __call__(self, engine: Engine, logger: TensorboardLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError("Handler 'GradsHistHandler' works only with TensorboardLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        for name, p in self.weights:
            if p.grad is None:
                continue

            name = name.replace(".", "/")
            logger.writer.add_histogram(
                tag=f"{tag_prefix}grads/{name}", values=p.grad.cpu().numpy(), global_step=global_step
            )
