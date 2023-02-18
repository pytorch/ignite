"""ClearML logger and its helper handlers."""
import os
import tempfile
import warnings
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, DefaultDict, List, Mapping, Optional, Tuple, Type, Union

from torch.optim import Optimizer

import ignite.distributed as idist
from ignite.contrib.handlers.base_logger import (
    BaseLogger,
    BaseOptimizerParamsHandler,
    BaseOutputHandler,
    BaseWeightsHandler,
    BaseWeightsScalarHandler,
)
from ignite.engine import Engine, Events
from ignite.handlers import global_step_from_engine
from ignite.handlers.checkpoint import DiskSaver

__all__ = [
    "ClearMLLogger",
    "ClearMLSaver",
    "OptimizerParamsHandler",
    "OutputHandler",
    "WeightsScalarHandler",
    "WeightsHistHandler",
    "GradsScalarHandler",
    "GradsHistHandler",
    "global_step_from_engine",
]


class ClearMLLogger(BaseLogger):
    """
    `ClearML <https://github.com/allegroai/clearml>`_ handler to log metrics, text, model/optimizer parameters,
    plots during training and validation.
    Also supports model checkpoints logging and upload to the storage solution of your choice (i.e. ClearML File server,
    S3 bucket etc.)

    .. code-block:: bash

        pip install clearml
        clearml-init

    Args:
        kwargs: Keyword arguments accepted from ``Task.init`` method.
            All arguments are optional. If a ClearML Task has already been created,
            kwargs will be ignored and the current ClearML Task will be used.

    Examples:
        .. code-block:: python

            from ignite.contrib.handlers.clearml_logger import *

            # Create a logger

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Attach the logger to the trainer to log training loss at each iteration
            clearml_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="training",
                output_transform=lambda loss: {"loss": loss}
            )

            # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
            # We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer` instead of `train_evaluator`.
            clearml_logger.attach_output_handler(
                train_evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="training",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer),
            )

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
            # `trainer` instead of `evaluator`.
            clearml_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer)),
            )

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            clearml_logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                optimizer=optimizer,
                param_name='lr'  # optional
            )

            # Attach the logger to the trainer to log model's weights norm after each iteration
            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(model)
            )

    """

    def __init__(self, **kwargs: Any):
        try:
            from clearml import Task
            from clearml.binding.frameworks.tensorflow_bind import WeightsGradientHistHelper
        except ImportError:
            raise ModuleNotFoundError(
                "This contrib module requires clearml to be installed. "
                "You may install clearml using: \n pip install clearml \n"
            )

        experiment_kwargs = {k: v for k, v in kwargs.items() if k not in ("project_name", "task_name", "task_type")}

        if self.bypass_mode():
            warnings.warn("ClearMLSaver: running in bypass mode")

            class _Stub(object):
                def __call__(self, *_: Any, **__: Any) -> "_Stub":
                    return self

                def __getattr__(self, attr: str) -> "_Stub":
                    if attr in ("name", "id"):
                        return ""  # type: ignore[return-value]
                    return self

                def __setattr__(self, attr: str, val: Any) -> None:
                    pass

            self._task = _Stub()
        else:
            # Try to retrieve current the ClearML Task before trying to create a new one
            self._task = Task.current_task()
            if self._task is None:
                self._task = Task.init(
                    project_name=kwargs.get("project_name"),
                    task_name=kwargs.get("task_name"),
                    task_type=kwargs.get("task_type", Task.TaskTypes.training),
                    **experiment_kwargs,
                )

        self.clearml_logger = self._task.get_logger()

        self.grad_helper = WeightsGradientHistHelper(logger=self.clearml_logger, report_freq=1)

    @classmethod
    def set_bypass_mode(cls, bypass: bool) -> None:
        """
        Will bypass all outside communication, and will drop all logs.
        Should only be used in "standalone mode", when there is no access to the *clearml-server*.

        Args:
            bypass: If ``True``, all outside communication is skipped.
        """
        setattr(cls, "_bypass", bypass)

    @classmethod
    def bypass_mode(cls) -> bool:
        """
        Returns the bypass mode state.

        Note:
            `GITHUB_ACTIONS` env will automatically set bypass_mode to ``True``
            unless overridden specifically with ``ClearMLLogger.set_bypass_mode(False)``.

        Return:
            If True, all outside communication is skipped.
        """
        return getattr(cls, "_bypass", bool(os.environ.get("CI")))

    def close(self) -> None:
        self.clearml_logger.flush()

    def _create_output_handler(self, *args: Any, **kwargs: Any) -> "OutputHandler":
        return OutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args: Any, **kwargs: Any) -> "OptimizerParamsHandler":
        return OptimizerParamsHandler(*args, **kwargs)


class OutputHandler(BaseOutputHandler):
    """Helper handler to log engine's output and/or metrics

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
            :meth:`~ignite.contrib.handlers.clearml_logger.global_step_from_engine`.
        state_attributes: list of attributes of the ``trainer.state`` to plot.

    Examples:
        .. code-block:: python

            from ignite.contrib.handlers.clearml_logger import *

            # Create a logger

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer`:
            clearml_logger.attach(
                evaluator,
                log_handler=OutputHandler(
                    tag="validation",
                    metric_names=["nll", "accuracy"],
                    global_step_transform=global_step_from_engine(trainer)
                ),
                event_name=Events.EPOCH_COMPLETED
            )
            # or equivalently
            clearml_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer)
            )

        Another example, where model is evaluated every 500 iterations:

        .. code-block:: python

            from ignite.contrib.handlers.clearml_logger import *

            @trainer.on(Events.ITERATION_COMPLETED(every=500))
            def evaluate(engine):
                evaluator.run(validation_set, max_epochs=1)

            # Create a logger

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            def global_step_transform(*args, **kwargs):
                return trainer.state.iteration

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # every 500 iterations. Since evaluator engine does not have access to the training iteration, we
            # provide a global_step_transform to return the trainer.state.iteration for the global_step, each time
            # evaluator metrics are plotted on ClearML.

            clearml_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metrics=["nll", "accuracy"],
                global_step_transform=global_step_transform
            )

        Another example where the State Attributes ``trainer.state.alpha`` and ``trainer.state.beta``
        are also logged along with the NLL and Accuracy after each iteration:

        .. code-block:: python

            clearml_logger.attach(
                trainer,
                log_handler=OutputHandler(
                    tag="training",
                    metric_names=["nll", "accuracy"],
                    state_attributes=["alpha", "beta"],
                ),
                event_name=Events.ITERATION_COMPLETED
            )

        Example of `global_step_transform`

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

    def __call__(self, engine: Engine, logger: ClearMLLogger, event_name: Union[str, Events]) -> None:

        if not isinstance(logger, ClearMLLogger):
            raise RuntimeError("Handler OutputHandler works only with ClearMLLogger")

        metrics = self._setup_output_metrics_state_attrs(engine)

        global_step = self.global_step_transform(engine, event_name)

        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}."
                " Please check the output of global_step_transform."
            )

        for key, value in metrics.items():
            if len(key) == 2:
                logger.clearml_logger.report_scalar(title=key[0], series=key[1], iteration=global_step, value=value)
            elif len(key) == 3:
                logger.clearml_logger.report_scalar(
                    title=f"{key[0]}/{key[1]}", series=key[2], iteration=global_step, value=value
                )


class OptimizerParamsHandler(BaseOptimizerParamsHandler):
    """Helper handler to log optimizer parameters

    Args:
        optimizer: torch optimizer or any object with attribute ``param_groups``
            as a sequence.
        param_name: parameter name
        tag: common title for all produced plots. For example, "generator"

    Examples:
        .. code-block:: python

            from ignite.contrib.handlers.clearml_logger import *

            # Create a logger

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            clearml_logger.attach(
                trainer,
                log_handler=OptimizerParamsHandler(optimizer),
                event_name=Events.ITERATION_STARTED
            )
            # or equivalently
            clearml_logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                optimizer=optimizer
            )
    """

    def __init__(self, optimizer: Optimizer, param_name: str = "lr", tag: Optional[str] = None):
        super(OptimizerParamsHandler, self).__init__(optimizer, param_name, tag)

    def __call__(self, engine: Engine, logger: ClearMLLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, ClearMLLogger):
            raise RuntimeError("Handler OptimizerParamsHandler works only with ClearMLLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        params = {
            str(i): float(param_group[self.param_name]) for i, param_group in enumerate(self.optimizer.param_groups)
        }

        for k, v in params.items():
            logger.clearml_logger.report_scalar(
                title=f"{tag_prefix}{self.param_name}", series=k, value=v, iteration=global_step
            )


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

            from ignite.contrib.handlers.clearml_logger import *

            # Create a logger

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Attach the logger to the trainer to log model's weights norm after each iteration
            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(model, reduction=torch.norm)
            )

        .. code-block:: python

            from ignite.contrib.handlers.clearml_logger import *

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Log only `fc` weights
            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(
                    model,
                    whitelist=['fc']
                )
            )

        .. code-block:: python

            from ignite.contrib.handlers.clearml_logger import *

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Log weights which have `bias` in their names
            def has_bias_in_name(n, p):
                return 'bias' in n

            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(model, whitelist=has_bias_in_name)
            )

    ..  versionchanged:: 0.4.9
        optional argument `whitelist` added.
    """

    def __call__(self, engine: Engine, logger: ClearMLLogger, event_name: Union[str, Events]) -> None:

        if not isinstance(logger, ClearMLLogger):
            raise RuntimeError("Handler WeightsScalarHandler works only with ClearMLLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        for name, p in self.weights:

            title_name, _, series_name = name.partition(".")
            logger.clearml_logger.report_scalar(
                title=f"{tag_prefix}weights_{self.reduction.__name__}/{title_name}",
                series=series_name,
                value=self.reduction(p.data),
                iteration=global_step,
            )


class WeightsHistHandler(BaseWeightsHandler):
    """Helper handler to log model's weights as histograms.

    Args:
        model: model to log weights
        tag: common title for all produced plots. For example, 'generator'
        whitelist: specific weights to log. Should be list of model's submodules
            or parameters names, or a callable which gets weight along with its name
            and determines if it should be logged. Names should be fully-qualified.
            For more information please refer to `PyTorch docs
            <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_submodule>`_.
            If not given, all of model's weights are logged.

    Examples:
        .. code-block:: python

            from ignite.contrib.handlers.clearml_logger import *

            # Create a logger

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Attach the logger to the trainer to log model's weights norm after each iteration
            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsHistHandler(model)
            )

        .. code-block:: python

            from ignite.contrib.handlers.clearml_logger import *

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Log weights of `fc` layer
            weights = ['fc']

            # Attach the logger to the trainer to log weights norm after each iteration
            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsHistHandler(model, whitelist=weights)
            )

        .. code-block:: python

            from ignite.contrib.handlers.clearml_logger import *

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Log weights which name include 'conv'.
            weight_selector = lambda name, p: 'conv' in name

            # Attach the logger to the trainer to log weights norm after each iteration
            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsHistHandler(model, whitelist=weight_selector)
            )

    ..  versionchanged:: 0.4.9
        optional argument `whitelist` added.
    """

    def __call__(self, engine: Engine, logger: ClearMLLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, ClearMLLogger):
            raise RuntimeError("Handler 'WeightsHistHandler' works only with ClearMLLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        for name, p in self.weights:

            title_name, _, series_name = name.partition(".")

            logger.grad_helper.add_histogram(
                title=f"{tag_prefix}weights_{title_name}",
                series=series_name,
                step=global_step,
                hist_data=p.data.cpu().numpy(),
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

            from ignite.contrib.handlers.clearml_logger import *

            # Create a logger

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Attach the logger to the trainer to log model's weights norm after each iteration
            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsScalarHandler(model, reduction=torch.norm)
            )

        .. code-block:: python

            from ignite.contrib.handlers.clearml_logger import *

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Log gradient of `base`
            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsScalarHandler(
                    model,
                    reduction=torch.norm,
                    whitelist=['base']
                )
            )

        .. code-block:: python

            from ignite.contrib.handlers.clearml_logger import *

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Log gradient of weights which belong to a `fc` layer
            def is_in_fc_layer(n, p):
                return 'fc' in n

            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsScalarHandler(model, whitelist=is_in_fc_layer)
            )

    ..  versionchanged:: 0.4.9
        optional argument `whitelist` added.
    """

    def __call__(self, engine: Engine, logger: ClearMLLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, ClearMLLogger):
            raise RuntimeError("Handler GradsScalarHandler works only with ClearMLLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        for name, p in self.weights:
            if p.grad is None:
                continue

            title_name, _, series_name = name.partition(".")
            logger.clearml_logger.report_scalar(
                title=f"{tag_prefix}grads_{self.reduction.__name__}/{title_name}",
                series=series_name,
                value=self.reduction(p.grad),
                iteration=global_step,
            )


class GradsHistHandler(BaseWeightsHandler):
    """Helper handler to log model's gradients as histograms.

    Args:
        model: model to log weights
        tag: common title for all produced plots. For example, 'generator'
        whitelist: specific gradients to log. Should be list of model's submodules
            or parameters names, or a callable which gets weight along with its name
            and determines if its gradient should be logged. Names should be
            fully-qualified. For more information please refer to `PyTorch docs
            <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_submodule>`_.
            If not given, all of model's gradients are logged.

    Examples:
        .. code-block:: python

            from ignite.contrib.handlers.clearml_logger import *

            # Create a logger

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Attach the logger to the trainer to log model's weights norm after each iteration
            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsHistHandler(model)
            )

        .. code-block:: python

            from ignite.contrib.handlers.clearml_logger import *

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Log gradient of `fc.bias`
            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsHistHandler(model, whitelist=['fc.bias'])
            )

        .. code-block:: python

            from ignite.contrib.handlers.clearml_logger import *

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            # Log gradient of weights which have shape (2, 1)
            def has_shape_2_1(n, p):
                return p.shape == (2,1)

            clearml_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsHistHandler(model, whitelist=has_shape_2_1)
            )

    ..  versionchanged:: 0.4.9
            optional argument `whitelist` added.
    """

    def __call__(self, engine: Engine, logger: ClearMLLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, ClearMLLogger):
            raise RuntimeError("Handler 'GradsHistHandler' works only with ClearMLLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        for name, p in self.weights:
            if p.grad is None:
                continue

            title_name, _, series_name = name.partition(".")
            logger.grad_helper.add_histogram(
                title=f"{tag_prefix}grads_{title_name}",
                series=series_name,
                step=global_step,
                hist_data=p.grad.cpu().numpy(),
            )


class ClearMLSaver(DiskSaver):
    """
    Handler that saves input checkpoint as ClearML artifacts

    Args:
        logger: An instance of :class:`~ignite.contrib.handlers.clearml_logger.ClearMLLogger`,
            ensuring a valid ClearML ``Task`` has been initialized. If not provided, and a ClearML Task
            has not been manually initialized, a runtime error will be raised.
        output_uri: The default location for output models and other artifacts uploaded by ClearML. For
            more information, see ``clearml.Task.init``.
        dirname: Directory path where the checkpoint will be saved. If not provided, a temporary
            directory will be created.

    Examples:
        .. code-block:: python

            from ignite.contrib.handlers.clearml_logger import *
            from ignite.handlers import Checkpoint

            clearml_logger = ClearMLLogger(
                project_name="pytorch-ignite-integration",
                task_name="cnn-mnist"
            )

            to_save = {"model": model}

            handler = Checkpoint(
                to_save,
                ClearMLSaver(),
                n_saved=1,
                score_function=lambda e: 123,
                score_name="acc",
                filename_prefix="best",
                global_step_transform=global_step_from_engine(trainer)
            )

            validation_evaluator.add_event_handler(Events.EVENT_COMPLETED, handler)

    """

    def __init__(
        self,
        logger: Optional[ClearMLLogger] = None,
        output_uri: Optional[str] = None,
        dirname: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ):

        self._setup_check_clearml(logger, output_uri)

        if not dirname:
            dirname = ""
            if idist.get_rank() == 0:
                dirname = tempfile.mkdtemp(prefix=f"ignite_checkpoints_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S_')}")
            if idist.get_world_size() > 1:
                dirname = idist.all_gather(dirname)[0]  # type: ignore[index, assignment]

            warnings.warn(f"ClearMLSaver created a temporary checkpoints directory: {dirname}")
            idist.barrier()

        # Let's set non-atomic tmp dir saving behaviour
        if "atomic" not in kwargs:
            kwargs["atomic"] = False

        self._checkpoint_slots: DefaultDict[Union[str, Tuple[str, str]], List[Any]] = defaultdict(list)

        super(ClearMLSaver, self).__init__(dirname=dirname, *args, **kwargs)  # type: ignore[misc]

    @idist.one_rank_only()
    def _setup_check_clearml(self, logger: ClearMLLogger, output_uri: str) -> None:
        try:
            from clearml import Task
        except ImportError:
            try:
                # Backwards-compatibility for legacy Trains SDK
                from trains import Task
            except ImportError:
                raise ModuleNotFoundError(
                    "This contrib module requires clearml to be installed. "
                    "You may install clearml using: \n pip install clearml \n"
                )

        if logger and not isinstance(logger, ClearMLLogger):
            raise TypeError("logger must be an instance of ClearMLLogger")

        self._task = Task.current_task()
        if not self._task:
            raise RuntimeError(
                "ClearMLSaver requires a ClearML Task to be initialized. "
                "Please use the `logger` argument or call `clearml.Task.init()`."
            )

        if output_uri:
            self._task.output_uri = output_uri

    class _CallbacksContext:
        def __init__(
            self,
            callback_type: Type[Enum],
            slots: List,
            checkpoint_key: str,
            filename: str,
            basename: str,
            metadata: Optional[Mapping] = None,
        ) -> None:
            self._callback_type = callback_type
            self._slots = slots
            self._checkpoint_key = str(checkpoint_key)
            self._filename = filename
            self._basename = basename
            self._metadata = metadata

        def pre_callback(self, action: str, model_info: Any) -> Any:
            if action != self._callback_type.save:  # type: ignore[attr-defined]
                return model_info

            try:
                slot = self._slots.index(None)
                self._slots[slot] = model_info.upload_filename
            except ValueError:
                self._slots.append(model_info.upload_filename)
                slot = len(self._slots) - 1

            model_info.upload_filename = f"{self._basename}_{slot}{os.path.splitext(self._filename)[1]}"
            model_info.local_model_id = f"{self._checkpoint_key}:{model_info.upload_filename}"
            return model_info

        def post_callback(self, action: str, model_info: Any) -> Any:
            if action != self._callback_type.save:  # type: ignore[attr-defined]
                return model_info

            model_info.model.name = f"{model_info.task.name}: {self._filename}"
            prefix = "Checkpoint Metadata: "
            metadata_items = ", ".join(f"{k}={v}" for k, v in self._metadata.items()) if self._metadata else "none"
            metadata = f"{prefix}{metadata_items}"
            comment = "\n".join(
                metadata if line.startswith(prefix) else line for line in (model_info.model.comment or "").split("\n")
            )
            if prefix not in comment:
                comment += "\n" + metadata
            model_info.model.comment = comment

            return model_info

    def __call__(self, checkpoint: Mapping, filename: str, metadata: Optional[Mapping] = None) -> None:
        try:
            from clearml.binding.frameworks import WeightsFileHandler
        except ImportError:
            try:
                # Backwards-compatibility for legacy Trains SDK
                from trains.binding.frameworks import WeightsFileHandler
            except ImportError:
                raise ModuleNotFoundError(
                    "This contrib module requires clearml to be installed. "
                    "You may install clearml using: \n pip install clearml \n"
                )

        try:
            basename = metadata["basename"]  # type: ignore[index]
        except (TypeError, KeyError):
            warnings.warn("Checkpoint metadata missing or basename cannot be found")
            basename = "checkpoint"

        checkpoint_key = (str(self.dirname), basename)

        cb_context = self._CallbacksContext(
            callback_type=WeightsFileHandler.CallbackType,
            slots=self._checkpoint_slots[checkpoint_key],
            checkpoint_key=str(checkpoint_key),
            filename=filename,
            basename=basename,
            metadata=metadata,
        )

        pre_cb_id = WeightsFileHandler.add_pre_callback(cb_context.pre_callback)
        post_cb_id = WeightsFileHandler.add_post_callback(cb_context.post_callback)

        try:
            super(ClearMLSaver, self).__call__(checkpoint, filename, metadata)
        finally:
            WeightsFileHandler.remove_pre_callback(pre_cb_id)
            WeightsFileHandler.remove_post_callback(post_cb_id)

    @idist.one_rank_only()
    def get_local_copy(self, filename: str) -> Optional[str]:
        """Get artifact local copy.

        .. warning::

            In distributed configuration this method should be called on rank 0 process.

        Args:
            filename: artifact name.

        Returns:
             a local path to a downloaded copy of the artifact
        """
        artifact = self._task.artifacts.get(filename)
        if artifact:
            return artifact.get_local_copy()
        self._task.get_logger().report_text(f"Can not find artifact {filename}")

        return None

    @idist.one_rank_only()
    def remove(self, filename: str) -> None:
        super(ClearMLSaver, self).remove(filename)
        for slots in self._checkpoint_slots.values():
            try:
                slots[slots.index(filename)] = None
            except ValueError:
                pass
            else:
                break
