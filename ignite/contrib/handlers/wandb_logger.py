from ignite.contrib.handlers.base_logger import BaseLogger, BaseOptimizerParamsHandler, BaseOutputHandler
from ignite.handlers import global_step_from_engine

__all__ = ["WandBLogger", "OutputHandler", "OptimizerParamsHandler", "global_step_from_engine"]


class OutputHandler(BaseOutputHandler):
    """Helper handler to log engine's output and/or metrics

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.wandb_logger import *

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

            from ignite.contrib.handlers.wandb_logger import *

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
            :meth:`~ignite.contrib.handlers.wandb_logger.global_step_from_engine`.
        sync (bool, optional): If set to False, process calls to log in a seperate thread. Default (None) uses whatever
            the default value of wandb.log.

    Note:

        Example of `global_step_transform`:

        .. code-block:: python

            def global_step_transform(engine, event_name):
                return engine.state.get_event_attrib_value(event_name)

    """

    def __init__(self, tag, metric_names=None, output_transform=None, global_step_transform=None, sync=None):
        super().__init__(tag, metric_names, output_transform, global_step_transform)
        self.sync = sync

    def __call__(self, engine, logger, event_name):

        if not isinstance(logger, WandBLogger):
            raise RuntimeError("Handler '{}' works only with WandBLogger.".format(self.__class__.__name__))

        global_step = self.global_step_transform(engine, event_name)
        if not isinstance(global_step, int):
            raise TypeError(
                "global_step must be int, got {}."
                " Please check the output of global_step_transform.".format(type(global_step))
            )

        metrics = self._setup_output_metrics(engine)
        if self.tag is not None:
            metrics = {"{tag}/{name}".format(tag=self.tag, name=name): value for name, value in metrics.items()}

        logger.log(metrics, step=global_step, sync=self.sync)


class OptimizerParamsHandler(BaseOptimizerParamsHandler):
    """Helper handler to log optimizer parameters

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.wandb_logger import *

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

    Args:
        optimizer (torch.optim.Optimizer): torch optimizer which parameters to log
        param_name (str): parameter name
        tag (str, optional): common title for all produced plots. For example, "generator"
        sync (bool, optional): If set to False, process calls to log in a seperate thread. Default (None) uses whatever
            the default value of wandb.log.
    """

    def __init__(self, optimizer, param_name="lr", tag=None, sync=None):
        super(OptimizerParamsHandler, self).__init__(optimizer, param_name, tag)
        self.sync = sync

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, WandBLogger):
            raise RuntimeError("Handler OptimizerParamsHandler works only with WandBLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        params = {
            "{}{}/group_{}".format(tag_prefix, self.param_name, i): float(param_group[self.param_name])
            for i, param_group in enumerate(self.optimizer.param_groups)
        }
        logger.log(params, step=global_step, sync=self.sync)


class WandBLogger(BaseLogger):
    """`Weights & Biases <https://app.wandb.ai/>`_ handler to log metrics, model/optimizer parameters, gradients
    during training and validation. It can also be used to log model checkpoints to the Weights & Biases cloud.

    .. code-block:: bash

        pip install wandb

    This class is also a wrapper for the wandb module. This means that you can call any wandb function using
    this wrapper. See examples on how to save model parameters and gradients.

    Args:
        *args: Positional arguments accepted by `wandb.init`.
        **kwargs: Keyword arguments accepted by `wandb.init`.
            Please see `wandb.init <https://docs.wandb.com/library/init>`_ for documentation of possible parameters.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.wandb_logger import *

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

    def __init__(self, *args, **kwargs):
        try:
            import wandb

            self._wandb = wandb
        except ImportError:
            raise RuntimeError(
                "This contrib module requires wandb to be installed. "
                "You man install wandb with the command:\n pip install wandb\n"
            )
        if kwargs.get("init", True):
            wandb.init(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self._wandb, attr)

    def _create_output_handler(self, *args, **kwargs):
        return OutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args, **kwargs):
        return OptimizerParamsHandler(*args, **kwargs)
