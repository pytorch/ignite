"""Neptune logger and its helper handlers."""
import tempfile
import warnings
from typing import Any, Callable, List, Mapping, Optional, Union

import torch
from torch.optim import Optimizer

import ignite.distributed as idist
from ignite import __version__
from ignite.contrib.handlers.base_logger import (
    BaseLogger,
    BaseOptimizerParamsHandler,
    BaseOutputHandler,
    BaseWeightsScalarHandler,
)
from ignite.engine import Engine, Events
from ignite.handlers import global_step_from_engine
from ignite.handlers.checkpoint import BaseSaveHandler

__all__ = [
    "NeptuneLogger",
    "NeptuneSaver",
    "OptimizerParamsHandler",
    "OutputHandler",
    "WeightsScalarHandler",
    "GradsScalarHandler",
    "global_step_from_engine",
]

_INTEGRATION_VERSION_KEY = "source_code/integrations/neptune-pytorch-ignite"


class NeptuneLogger(BaseLogger):
    """
    `Neptune <https://neptune.ai/>`_ handler to log metrics, model/optimizer parameters and gradients during training
    and validation. It can also log model checkpoints to Neptune.

    .. code-block:: bash

        pip install neptune

    Args:
        api_token: Neptune API token, found on https://neptune.ai -> User menu -> "Get your API token".
           If None, the value of the NEPTUNE_API_TOKEN environment variable is used. To keep your token
           secure, you should set it to the environment variable rather than including it in your code.
        project: Name of a Neptune project, in the form "workspace-name/project-name".
           For example "tom/mnist-classification".
           If None, the value of the NEPTUNE_PROJECT environment variable is used.
        **kwargs: Other arguments to be passed to the `init_run()` function.

    Examples:
        .. code-block:: python

            from ignite.contrib.handlers.neptune_logger import *

            # Create a logger
            # Note: We are using the API token for anonymous logging. You can pass your own token, or save it as an
            # environment variable and leave out the api_token argument.

            npt_logger = NeptuneLogger(
                api_token="ANONYMOUS",
                project="common/pytorch-ignite-integration",
                name="cnn-mnist",  # Optional,
                tags=["pytorch-ignite", "minst"],  # Optional
            )

            # Attach the logger to the trainer to log training loss at each iteration.
            npt_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="training",
                output_transform=lambda loss: {"loss": loss},
            )

            # Attach the logger to the evaluator on the training dataset and log NLL
            # and accuracy metrics after each epoch.
            # We set up `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer` instead of `train_evaluator`.
            npt_logger.attach_output_handler(
                train_evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="training",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer),
            )

            # Attach the logger to the evaluator on the validation dataset and log NLL and accuracy metrics after
            # each epoch. We set up `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
            # `trainer` instead of `evaluator`.
            npt_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer),
            )

            # Attach the logger to the trainer to log optimizer parameters, such as learning rate at each iteration.
            npt_logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                optimizer=optimizer,
                param_name="lr",  # optional
            )

            # Attach the logger to the trainer to log model's weights norm after each iteration.
            npt_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(model),
            )

        Explore runs with Neptune tracking here:
        https://app.neptune.ai/o/common/org/pytorch-ignite-integration/

        You can also save model checkpoints to a Neptune:

        .. code-block:: python

            from ignite.handlers import Checkpoint


            def score_function(engine):
                return engine.state.metrics["accuracy"]


            to_save = {"model": model}
            handler = Checkpoint(
                to_save,
                NeptuneSaver(npt_logger), n_saved=2,
                filename_prefix="best",
                score_function=score_function,
                score_name="validation_accuracy",
                global_step_transform=global_step_from_engine(trainer),
            )
            validation_evaluator.add_event_handler(Events.COMPLETED, handler)

        It is also possible to use the logger as a context manager:

        .. code-block:: python

            from ignite.contrib.handlers.neptune_logger import *

            with NeptuneLogger() as npt_logger:
                trainer = Engine(update_fn)
                # Attach the logger to the trainer to log training loss at each iteration
                npt_logger.attach_output_handler(
                    trainer,
                    event_name=Events.ITERATION_COMPLETED,
                    tag="training",
                    output_transform=lambda loss: {"loss": loss},
                )

    """

    def __getattr__(self, attr: Any) -> Any:
        return getattr(self.experiment, attr)

    def __getitem__(self, key: str) -> Any:
        return self.experiment[key]

    def __setitem__(self, key: str, val: Any) -> Any:
        self.experiment[key] = val

    def __init__(self, api_token: Optional[str] = None, project: Optional[str] = None, **kwargs: Any) -> None:
        try:
            try:
                # neptune-client<1.0.0 package structure
                with warnings.catch_warnings():
                    # ignore the deprecation warnings
                    warnings.simplefilter("ignore")
                    import neptune.new as neptune
            except ImportError:
                # neptune>=1.0.0 package structure
                import neptune
        except ImportError:
            raise ModuleNotFoundError(
                "This contrib module requires the Neptune client library to be installed. "
                "Install neptune with the command: \n pip install neptune \n"
            )

        run = neptune.init_run(
            api_token=api_token,
            project=project,
            **kwargs,
        )
        run[_INTEGRATION_VERSION_KEY] = __version__

        self.experiment = run

    def close(self) -> None:
        self.experiment.stop()

    def _create_output_handler(self, *args: Any, **kwargs: Any) -> "OutputHandler":
        return OutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args: Any, **kwargs: Any) -> "OptimizerParamsHandler":
        return OptimizerParamsHandler(*args, **kwargs)


class OutputHandler(BaseOutputHandler):
    """Helper handler to log engine's output and/or metrics.

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
            :meth:`~ignite.contrib.handlers.neptune_logger.global_step_from_engine`.
        state_attributes: list of attributes of the ``trainer.state`` to plot.

    Examples:
        .. code-block:: python

            from ignite.contrib.handlers.neptune_logger import *

            # Create a logger
            # We are using the api_token for the anonymous user neptuner but you can use your own.

            npt_logger = NeptuneLogger(
                api_token="ANONYMOUS",
                project_name="shared/pytorch-ignite-integration",
                experiment_name="cnn-mnist", # Optional,
                params={"max_epochs": 10}, # Optional,
                tags=["pytorch-ignite","minst"] # Optional
            )

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer`:
            npt_logger.attach(
                evaluator,
                log_handler=OutputHandler(
                    tag="validation",
                    metric_names=["nll", "accuracy"],
                    global_step_transform=global_step_from_engine(trainer)
                ),
                event_name=Events.EPOCH_COMPLETED
            )
            # or equivalently
            npt_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer)
            )

        Another example, where model is evaluated every 500 iterations:

        .. code-block:: python

            from ignite.contrib.handlers.neptune_logger import *

            @trainer.on(Events.ITERATION_COMPLETED(every=500))
            def evaluate(engine):
                evaluator.run(validation_set, max_epochs=1)

            # We are using the api_token for the anonymous user neptuner but you can use your own.

            npt_logger = NeptuneLogger(
                api_token="ANONYMOUS",
                project_name="shared/pytorch-ignite-integration",
                experiment_name="cnn-mnist", # Optional,
                params={"max_epochs": 10}, # Optional,
                tags=["pytorch-ignite", "minst"] # Optional
            )

            def global_step_transform(*args, **kwargs):
                return trainer.state.iteration

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # every 500 iterations. Since evaluator engine does not have access to the training iteration, we
            # provide a global_step_transform to return the trainer.state.iteration for the global_step, each time
            # evaluator metrics are plotted on NeptuneML.

            npt_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metrics=["nll", "accuracy"],
                global_step_transform=global_step_transform
            )

        Another example where the State Attributes ``trainer.state.alpha`` and ``trainer.state.beta``
        are also logged along with the NLL and Accuracy after each iteration:

        .. code-block:: python

            npt_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="training",
                metrics=["nll", "accuracy"],
                state_attributes=["alpha", "beta"],
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
        metric_names: Optional[Union[str, List[str]]] = None,
        output_transform: Optional[Callable] = None,
        global_step_transform: Optional[Callable[[Engine, Union[str, Events]], int]] = None,
        state_attributes: Optional[List[str]] = None,
    ):
        super(OutputHandler, self).__init__(
            tag, metric_names, output_transform, global_step_transform, state_attributes
        )

    def __call__(self, engine: Engine, logger: NeptuneLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, NeptuneLogger):
            raise TypeError("Handler OutputHandler works only with NeptuneLogger")

        metrics = self._setup_output_metrics_state_attrs(engine, key_tuple=False)

        global_step = self.global_step_transform(engine, event_name)

        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}."
                " Please check the output of global_step_transform."
            )

        for key, value in metrics.items():
            logger[key].append(value, step=global_step)


class OptimizerParamsHandler(BaseOptimizerParamsHandler):
    """Helper handler to log optimizer parameters

    Args:
        optimizer: torch optimizer or any object with attribute ``param_groups``
            as a sequence.
        param_name: parameter name
        tag: common title for all produced plots. For example, "generator"

    Examples:
        .. code-block:: python

            from ignite.contrib.handlers.neptune_logger import *

            # Create a logger
            # We are using the api_token for the anonymous user neptuner but you can use your own.

            npt_logger = NeptuneLogger(
                api_token="ANONYMOUS",
                project_name="shared/pytorch-ignite-integration",
                experiment_name="cnn-mnist", # Optional,
                params={"max_epochs": 10}, # Optional,
                tags=["pytorch-ignite","minst"] # Optional
            )

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            npt_logger.attach(
                trainer,
                log_handler=OptimizerParamsHandler(optimizer),
                event_name=Events.ITERATION_STARTED
            )
            # or equivalently
            npt_logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                optimizer=optimizer
            )
    """

    def __init__(self, optimizer: Optimizer, param_name: str = "lr", tag: Optional[str] = None):
        super(OptimizerParamsHandler, self).__init__(optimizer, param_name, tag)

    def __call__(self, engine: Engine, logger: NeptuneLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, NeptuneLogger):
            raise TypeError("Handler OptimizerParamsHandler works only with NeptuneLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        params = {
            f"{tag_prefix}{self.param_name}/group_{i}": float(param_group[self.param_name])
            for i, param_group in enumerate(self.optimizer.param_groups)
        }

        for k, v in params.items():
            logger[k].append(v, step=global_step)


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

            from ignite.contrib.handlers.neptune_logger import *

            # Create a logger
            # We are using the api_token for the anonymous user neptuner but you can use your own.

            npt_logger = NeptuneLogger(
                api_token="ANONYMOUS",
                project_name="shared/pytorch-ignite-integration",
                experiment_name="cnn-mnist", # Optional,
                params={"max_epochs": 10}, # Optional,
                tags=["pytorch-ignite","minst"] # Optional
            )

            # Attach the logger to the trainer to log model's weights norm after each iteration
            npt_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(model, reduction=torch.norm)
            )

        .. code-block:: python

            from ignite.contrib.handlers.neptune_logger import *

            npt_logger = NeptuneLogger(
                api_token="ANONYMOUS",
                project_name="shared/pytorch-ignite-integration",
                experiment_name="cnn-mnist", # Optional,
                params={"max_epochs": 10}, # Optional,
                tags=["pytorch-ignite","minst"] # Optional
            )

            # Log only `fc` weights
            npt_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(
                    model,
                    whitelist=['fc']
                )
            )

        .. code-block:: python

            from ignite.contrib.handlers.neptune_logger import *

            npt_logger = NeptuneLogger(
                api_token="ANONYMOUS",
                project_name="shared/pytorch-ignite-integration",
                experiment_name="cnn-mnist", # Optional,
                params={"max_epochs": 10}, # Optional,
                tags=["pytorch-ignite","minst"] # Optional
            )

            # Log weights which have `bias` in their names
            def has_bias_in_name(n, p):
                return 'bias' in n

            npt_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(model, whitelist=has_bias_in_name)
            )

    ..  versionchanged:: 0.4.9
        optional argument `whitelist` added.
    """

    def __call__(self, engine: Engine, logger: NeptuneLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, NeptuneLogger):
            raise TypeError("Handler WeightsScalarHandler works only with NeptuneLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        for name, p in self.weights:
            if p.grad is None:
                continue

            name = name.replace(".", "/")
            key = f"{tag_prefix}weights_{self.reduction.__name__}/{name}"
            logger[key].append(self.reduction(p.data), step=global_step)


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

            from ignite.contrib.handlers.neptune_logger import *

            # Create a logger
            # We are using the api_token for the anonymous user neptuner but you can use your own.

            npt_logger = NeptuneLogger(
                api_token="ANONYMOUS",
                project_name="shared/pytorch-ignite-integration",
                experiment_name="cnn-mnist", # Optional,
                params={"max_epochs": 10}, # Optional,
                tags=["pytorch-ignite","minst"] # Optional
            )

            # Attach the logger to the trainer to log model's weights norm after each iteration
            npt_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsScalarHandler(model, reduction=torch.norm)
            )

        .. code-block:: python

            from ignite.contrib.handlers.neptune_logger import *

            npt_logger = NeptuneLogger(
                api_token="ANONYMOUS",
                project_name="shared/pytorch-ignite-integration",
                experiment_name="cnn-mnist", # Optional,
                params={"max_epochs": 10}, # Optional,
                tags=["pytorch-ignite","minst"] # Optional
            )

            # Log gradient of `base`
            npt_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsScalarHandler(
                    model,
                    reduction=torch.norm,
                    whitelist=['base']
                )
            )

        .. code-block:: python

            from ignite.contrib.handlers.neptune_logger import *

            npt_logger = NeptuneLogger(
                api_token="ANONYMOUS",
                project_name="shared/pytorch-ignite-integration",
                experiment_name="cnn-mnist", # Optional,
                params={"max_epochs": 10}, # Optional,
                tags=["pytorch-ignite","minst"] # Optional
            )

            # Log gradient of weights which belong to a `fc` layer
            def is_in_fc_layer(n, p):
                return 'fc' in n

            npt_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsScalarHandler(model, whitelist=is_in_fc_layer)
            )

    ..  versionchanged:: 0.4.9
        optional argument `whitelist` added.
    """

    def __call__(self, engine: Engine, logger: NeptuneLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, NeptuneLogger):
            raise TypeError("Handler GradsScalarHandler works only with NeptuneLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        for name, p in self.weights:
            if p.grad is None:
                continue

            name = name.replace(".", "/")
            key = f"{tag_prefix}grads_{self.reduction.__name__}/{name}"
            logger[key].append(self.reduction(p.grad), step=global_step)


class NeptuneSaver(BaseSaveHandler):
    """Handler that saves input checkpoint to the Neptune server.

    Args:
        neptune_logger: an instance of
            NeptuneLogger class.

    .. Note ::

        NeptuneSaver is currently not supported on Windows.

    Examples:
        .. code-block:: python

            from ignite.contrib.handlers.neptune_logger import *

            # Create a logger
            # We are using the api_token for the anonymous user neptuner but you can use your own.

            npt_logger = NeptuneLogger(
                api_token="ANONYMOUS",
                project_name="shared/pytorch-ignite-integration",
                experiment_name="cnn-mnist", # Optional,
                params={"max_epochs": 10}, # Optional,
                tags=["pytorch-ignite","minst"] # Optional
            )

            ...
            evaluator = create_supervised_evaluator(model, metrics=metrics, ...)
            ...

            from ignite.handlers import Checkpoint

            def score_function(engine):
                return engine.state.metrics["accuracy"]

            to_save = {"model": model}

            # pass neptune logger to NeptuneServer

            handler = Checkpoint(
                to_save,
                NeptuneSaver(npt_logger), n_saved=2,
                filename_prefix="best", score_function=score_function,
                score_name="validation_accuracy",
                global_step_transform=global_step_from_engine(trainer)
            )

            evaluator.add_event_handler(Events.COMPLETED, handler)

            # We need to close the logger when we are done
            npt_logger.close()

    For example, you can access model checkpoints and download them from here:
    https://ui.neptune.ai/o/shared/org/pytorch-ignite-integration/e/PYTOR1-18/charts

    """

    @idist.one_rank_only()
    def __init__(self, neptune_logger: NeptuneLogger):
        self._logger = neptune_logger

    @idist.one_rank_only()
    def __call__(self, checkpoint: Mapping, filename: str, metadata: Optional[Mapping] = None) -> None:
        # wont work on XLA

        # Imports for BC compatibility
        try:
            # neptune-client<1.0.0 package structure
            with warnings.catch_warnings():
                # ignore the deprecation warnings
                warnings.simplefilter("ignore")
                from neptune.new.types import File
        except ImportError:
            # neptune>=1.0.0 package structure
            from neptune.types import File

        with tempfile.NamedTemporaryFile() as tmp:
            # we can not use tmp.name to open tmp.file twice on Win32
            # https://docs.python.org/3/library/tempfile.html#tempfile.NamedTemporaryFile
            torch.save(checkpoint, tmp.file)

            # rewind the buffer
            tmp.file.seek(0)

            # hold onto the file stream for uploading.
            # NOTE: This won't load the whole file in memory and upload
            #       the stream in smaller chunks.
            self._logger[filename].upload(File.from_stream(tmp.file))

    @idist.one_rank_only(with_barrier=True)
    def remove(self, filename: str) -> None:
        del self._logger.experiment[filename]
