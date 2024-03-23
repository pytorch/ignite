"""WandB logger and its helper handlers."""

from typing import Any, Callable, List, Optional, Union

from torch.optim import Optimizer

from ignite.engine import Engine, Events

from ignite.handlers.base_logger import BaseLogger, BaseOptimizerParamsHandler, BaseOutputHandler
from ignite.handlers.utils import global_step_from_engine  # noqa

__all__ = ["WandBLogger", "OutputHandler", "OptimizerParamsHandler", "global_step_from_engine"]


class WandBLogger(BaseLogger):
    """`Weights & Biases <https://wandb.ai/site>`_ handler to log metrics, model/optimizer parameters, gradients
    during training and validation. It can also be used to log model checkpoints to the Weights & Biases cloud.

    .. code-block:: bash

        pip install wandb

    This class is also a wrapper for the wandb module. This means that you can call any wandb function using
    this wrapper. See examples on how to save model parameters and gradients.

    Args:
        args: Positional arguments accepted by `wandb.init`.
        kwargs: Keyword arguments accepted by `wandb.init`.
            Please see `wandb.init <https://docs.wandb.ai/library/init>`_ for documentation of possible parameters.

    Examples:
        .. code-block:: python

            from ignite.handlers.wandb_logger import *

            # Create a logger. All parameters are optional. See documentation
            # on wandb.init for details.

            wandb_logger = WandBLogger(
                entity="shared",
                project="pytorch-ignite-integration",
                name="cnn-mnist",
                config={"max_epochs": 10},
                tags=["pytorch-ignite", "minst"]
            )

            # Attach the logger to the trainer to log training loss at each iteration
            wandb_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="training",
                output_transform=lambda loss: {"loss": loss}
            )

            # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
            # We setup `global_step_transform=lambda *_: trainer.state.iteration` to take iteration value
            # of the `trainer`:
            wandb_logger.attach_output_handler(
                train_evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="training",
                metric_names=["nll", "accuracy"],
                global_step_transform=lambda *_: trainer.state.iteration,
            )

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=lambda *_: trainer.state.iteration` to take iteration value
            # of the `trainer` instead of `evaluator`.
            wandb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=["nll", "accuracy"],
                global_step_transform=lambda *_: trainer.state.iteration,
            )

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            wandb_logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                optimizer=optimizer,
                param_name='lr'  # optional
            )

            # We need to close the logger when we are done
            wandb_logger.close()

        If you want to log model gradients, the model call graph, etc., use the logger as wrapper of wandb. Refer
        to the documentation of wandb.watch for details:

        .. code-block:: python

            wandb_logger = WandBLogger(
                entity="shared",
                project="pytorch-ignite-integration",
                name="cnn-mnist",
                config={"max_epochs": 10},
                tags=["pytorch-ignite", "minst"]
            )

            model = torch.nn.Sequential(...)
            wandb_logger.watch(model)

        For model checkpointing, Weights & Biases creates a local run dir, and automatically synchronizes all
        files saved there at the end of the run. You can just use the `wandb_logger.run.dir` as path for the
        `ModelCheckpoint`:

        .. code-block:: python

            from ignite.handlers import ModelCheckpoint

            def score_function(engine):
                return engine.state.metrics['accuracy']

            model_checkpoint = ModelCheckpoint(
                wandb_logger.run.dir, n_saved=2, filename_prefix='best',
                require_empty=False, score_function=score_function,
                score_name="validation_accuracy",
                global_step_transform=global_step_from_engine(trainer)
            )
            evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {'model': model})


    """

    def __init__(self, *args: Any, **kwargs: Any):
        try:
            import wandb

            self._wandb = wandb
        except ImportError:
            raise ModuleNotFoundError(
                "This contrib module requires wandb to be installed. "
                "You man install wandb with the command:\n pip install wandb\n"
            )
        if kwargs.get("init", True):
            wandb.init(*args, **kwargs)

    def __getattr__(self, attr: Any) -> Any:
        return getattr(self._wandb, attr)

    def close(self) -> None:
        self._wandb.finish()

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
            :meth:`~ignite.handlers.wandb_logger.global_step_from_engine`.
        sync: If set to False, process calls to log in a seperate thread. Default (None) uses whatever
            the default value of wandb.log.

    Examples:
        .. code-block:: python

            from ignite.handlers.wandb_logger import *

            # Create a logger. All parameters are optional. See documentation
            # on wandb.init for details.

            wandb_logger = WandBLogger(
                entity="shared",
                project="pytorch-ignite-integration",
                name="cnn-mnist",
                config={"max_epochs": 10},
                tags=["pytorch-ignite", "minst"]
            )

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=lambda *_: trainer.state.iteration,` to take iteration value
            # of the `trainer`:
            wandb_logger.attach(
                evaluator,
                log_handler=OutputHandler(
                    tag="validation",
                    metric_names=["nll", "accuracy"],
                    global_step_transform=lambda *_: trainer.state.iteration,
                ),
                event_name=Events.EPOCH_COMPLETED
            )
            # or equivalently
            wandb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=["nll", "accuracy"],
                global_step_transform=lambda *_: trainer.state.iteration,
            )

        Another example, where model is evaluated every 500 iterations:

        .. code-block:: python

            from ignite.handlers.wandb_logger import *

            @trainer.on(Events.ITERATION_COMPLETED(every=500))
            def evaluate(engine):
                evaluator.run(validation_set, max_epochs=1)

            # Create a logger. All parameters are optional. See documentation
            # on wandb.init for details.

            wandb_logger = WandBLogger(
                entity="shared",
                project="pytorch-ignite-integration",
                name="cnn-mnist",
                config={"max_epochs": 10},
                tags=["pytorch-ignite", "minst"]
            )

            def global_step_transform(*args, **kwargs):
                return trainer.state.iteration

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # every 500 iterations. Since evaluator engine does not have access to the training iteration, we
            # provide a global_step_transform to return the trainer.state.iteration for the global_step, each time
            # evaluator metrics are plotted on Weights & Biases.

            wandb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metrics=["nll", "accuracy"],
                global_step_transform=global_step_transform
            )

        Another example where the State Attributes ``trainer.state.alpha`` and ``trainer.state.beta``
        are also logged along with the NLL and Accuracy after each iteration:

        .. code-block:: python

            wandb_logger.attach_output_handler(
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
        metric_names: Optional[List[str]] = None,
        output_transform: Optional[Callable] = None,
        global_step_transform: Optional[Callable[[Engine, Union[str, Events]], int]] = None,
        sync: Optional[bool] = None,
        state_attributes: Optional[List[str]] = None,
    ):
        super().__init__(tag, metric_names, output_transform, global_step_transform, state_attributes)
        self.sync = sync

    def __call__(self, engine: Engine, logger: WandBLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, WandBLogger):
            raise RuntimeError(f"Handler '{self.__class__.__name__}' works only with WandBLogger.")

        global_step = self.global_step_transform(engine, event_name)
        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}."
                " Please check the output of global_step_transform."
            )

        metrics = self._setup_output_metrics_state_attrs(engine, log_text=True, key_tuple=False)
        logger.log(metrics, step=global_step, sync=self.sync)


class OptimizerParamsHandler(BaseOptimizerParamsHandler):
    """Helper handler to log optimizer parameters

    Args:
        optimizer: torch optimizer or any object with attribute ``param_groups``
            as a sequence.
        param_name: parameter name
        tag: common title for all produced plots. For example, "generator"
        sync: If set to False, process calls to log in a seperate thread. Default (None) uses whatever
            the default value of wandb.log.

    Examples:
        .. code-block:: python

            from ignite.handlers.wandb_logger import *

            # Create a logger. All parameters are optional. See documentation
            # on wandb.init for details.

            wandb_logger = WandBLogger(
                entity="shared",
                project="pytorch-ignite-integration",
                name="cnn-mnist",
                config={"max_epochs": 10},
                tags=["pytorch-ignite", "minst"]
            )

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            wandb_logger.attach(
                trainer,
                log_handler=OptimizerParamsHandler(optimizer),
                event_name=Events.ITERATION_STARTED
            )
            # or equivalently
            wandb_logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                optimizer=optimizer
            )
    """

    def __init__(
        self, optimizer: Optimizer, param_name: str = "lr", tag: Optional[str] = None, sync: Optional[bool] = None
    ):
        super(OptimizerParamsHandler, self).__init__(optimizer, param_name, tag)
        self.sync = sync

    def __call__(self, engine: Engine, logger: WandBLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, WandBLogger):
            raise RuntimeError("Handler OptimizerParamsHandler works only with WandBLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        params = {
            f"{tag_prefix}{self.param_name}/group_{i}": float(param_group[self.param_name])
            for i, param_group in enumerate(self.optimizer.param_groups)
        }
        logger.log(params, step=global_step, sync=self.sync)
