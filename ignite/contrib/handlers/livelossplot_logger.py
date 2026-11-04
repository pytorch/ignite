import numbers
import warnings
from enum import Enum
from typing import Callable, Iterable, Optional, Union

import torch
from torch.optim.optimizer import Optimizer

from ignite.contrib.handlers.base_logger import BaseLogger, BaseOptimizerParamsHandler, BaseOutputHandler
from ignite.engine import Engine
from ignite.engine.events import CallableEventWithFilter
from ignite.handlers import global_step_from_engine

__all__ = [
    "LivelossplotLogger",
    "OptimizerParamsHandler",
    "OutputHandler",
    "global_step_from_engine",
]


class OutputHandler(BaseOutputHandler):
    """Helper handler to log engine's output and/or metrics

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.livelossplot_logger import *

            # Create a logger
            livelossplot_logger = LivelossplotLogger()

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer`:
            livelossplot_logger.attach(
                evaluator,
                log_handler=OutputHandler(
                    tag="validation",
                    metric_names=["nll", "accuracy"],
                    global_step_transform=global_step_from_engine(trainer)
                ),
                event_name=Events.EPOCH_COMPLETED
            )
            # or equivalently
            livelossplot_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer)
            )

        Another example, where model is evaluated every 500 iterations:

        .. code-block:: python

            from ignite.contrib.handlers.livelossplot_logger import *

            @trainer.on(Events.ITERATION_COMPLETED(every=500))
            def evaluate(engine):
                evaluator.run(validation_set, max_epochs=1)

            livelossplot_logger = LivelossplotLogger()

            def global_step_transform(*args, **kwargs):
                return trainer.state.iteration

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # every 500 iterations. Since evaluator engine does not have access to the training iteration, we
            # provide a global_step_transform to return the trainer.state.iteration for the global_step, each time
            # evaluator metrics are plotted on chosen outputs - like Matplotlib, Tensorboard, Neptune etc.

            livelossplot_logger.attach_output_handler(
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
            :meth:`~ignite.contrib.handlers.livelossplot_logger.global_step_from_engine`.

    Note:

        Example of `global_step_transform`:

        .. code-block:: python

            def global_step_transform(engine, event_name):
                return engine.state.get_event_attrib_value(event_name)

    """

    def __init__(
        self,
        tag: str,
        metric_names: Optional[Iterable[str]] = None,
        output_transform: Optional[Callable] = None,
        global_step_transform: Optional[Callable] = None,
        # drawing_handler: bool = False,
    ):
        super(OutputHandler, self).__init__(tag, metric_names, output_transform, global_step_transform)
        # self.drawing_handler = drawing_handler

    def __call__(self, engine, logger, event_name: str):

        if not isinstance(logger, LivelossplotLogger):
            raise RuntimeError("Handler 'OutputHandler' works only with LivelossplotLogger")

        metrics = self._setup_output_metrics(engine)

        global_step = self.global_step_transform(engine, event_name)
        if not isinstance(global_step, int):
            raise TypeError(
                "global_step must be int, got {}."
                " Please check the output of global_step_transform.".format(type(global_step))
            )

        logs = {}
        for key, value in metrics.items():
            if isinstance(value, numbers.Number):
                logs["{} {}".format(self.tag, key)] = value
            elif isinstance(value, torch.Tensor) and value.ndimension() == 0:
                logs["{} {}".format(self.tag, key)] = value.item()
            elif isinstance(value, torch.Tensor) and value.ndimension() == 1:
                for i, v in enumerate(value):
                    logs["{} {} {}".format(self.tag, key, i)] = v.item()
            else:
                warnings.warn(
                    "LivelossplotLogger output_handler can not log metrics value type {}".format(type(value))
                )

        logger.writer.update(logs, current_step=global_step)
        # if self.drawing_handler:
        logger.writer.send()


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
        optimizer (torch.optim.Optimizer): torch optimizer which parameters to log
        param_name (str): parameter name
        tag (str, optional): common title for all produced plots. For example, "generator"
    """

    def __init__(
        self, optimizer: Optimizer, param_name: str = "lr", tag: Optional[str] = None,  # drawing_handler: bool = False
    ):
        super(OptimizerParamsHandler, self).__init__(optimizer, param_name, tag)
        # self.drawing_handler = drawing_handler

    def __call__(self, engine: Engine, logger: BaseLogger, event_name: Union[CallableEventWithFilter, Enum]):
        if not isinstance(logger, LivelossplotLogger):
            raise RuntimeError("Handler OptimizerParamsHandler works only with LivelossplotLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        params = {
            "{}{}/group_{}".format(tag_prefix, self.param_name, i): float(param_group[self.param_name])
            for i, param_group in enumerate(self.optimizer.param_groups)
        }

        logger.writer.update(params, current_step=global_step)
        # if self.drawing_handler:
        logger.writer.send()


class LivelossplotLogger(BaseLogger):
    """
    Livelossplot handler to log metrics, model/optimizer running parameters during the training and validation.

    Args:
        **kwargs: Keyword arguments accepted from
            `PlotLosses
            <https://github.com/stared/livelossplot/blob/master/livelossplot/plot_losses.py>`_.
            For example, `group_patterns` to setup regex pattern for logs grouping.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.livelossplot_logger import *

            # Create a logger
            llp_logger = LivelossplotLogger()

            # Attach the logger to the trainer to log training loss at each iteration
            llp_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="online",
                output_transform=lambda loss: {"batch loss": loss}
            )

            # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
            # We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer` instead of `train_evaluator`.
            llp_logger.attach_output_handler(
                train_evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="training",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer),
            )

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
            # `trainer` instead of `evaluator`.
            llp_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer)),
            )

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            llp_logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                optimizer=optimizer,
                param_name='lr'  # optional
            )

        It is also possible to use the logger as context manager:

        .. code-block:: python

            from ignite.contrib.handlers.livelossplot_logger import *

            with LivelossplotLogger() as llp_logger:

                trainer = Engine(update_fn)
                # Attach the logger to the trainer to log training loss at each iteration
                llp_logger.attach_output_handler(
                    trainer,
                    event_name=Events.ITERATION_COMPLETED,
                    tag="training",
                    output_transform=lambda loss: {"loss": loss}
                )

    """

    def __init__(self, *args, **kwargs):

        try:
            import livelossplot
        except ImportError:
            raise RuntimeError(
                "This contrib module requires livelossplot to be installed. "
                "You may install trains using: \n pip install livelossplot \n"
            )

        from livelossplot import PlotLosses
        from livelossplot.outputs import MatplotlibPlot

        if "outputs" not in kwargs:
            kwargs["outputs"] = [MatplotlibPlot(), ]

        self.writer = PlotLosses(*args, **kwargs)
        # THIS IS AN ATTEMPT TO AVOID MULTIPLE CALLS OF .send() ON THE SAME EVENT
        # self._
        # self.drawing_handler_attached = False

    def _create_output_handler(self, *args, **kwargs):
        # if not self.drawing_handler_attached:
        #     self.drawing_handler_attached = True
        #     return OutputHandler(*args, drawing_handler=True, **kwargs)

        # Setup group_patterns based on tags if currently no group_patterns
        # group_patterns = [
        #     (r"^(training(\s))(.*)", "training "),
        #     (r"^(validation(\s))(.*)", "validation "),
        #     (r"^(online(\s)batch(\s))(.*)", "online batch "),
        # ]
        return OutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args, **kwargs):
        # if not self.drawing_handler_attached:
        #     self.drawing_handler_attached = True
        #     return OptimizerParamsHandler(*args, drawing_handler=True, **kwargs)
        return OptimizerParamsHandler(*args, **kwargs)
