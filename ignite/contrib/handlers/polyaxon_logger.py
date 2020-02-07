import numbers

import warnings
import torch

from ignite.contrib.handlers.base_logger import BaseLogger, BaseOutputHandler, BaseOptimizerParamsHandler, \
    global_step_from_engine

__all__ = ['PolyaxonLogger', 'OutputHandler', 'OptimizerParamsHandler', 'global_step_from_engine']


class OutputHandler(BaseOutputHandler):
    """Helper handler to log engine's output and/or metrics.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.polyaxon_logger import *

            # Create a logger
            plx_logger = PolyaxonLogger()

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer`:
            plx_logger.attach(evaluator,
                              log_handler=OutputHandler(tag="validation",
                                                        metric_names=["nll", "accuracy"],
                                                        global_step_transform=global_step_from_engine(trainer)),
                              event_name=Events.EPOCH_COMPLETED)

        Another example, where model is evaluated every 500 iterations:

        .. code-block:: python

            from ignite.contrib.handlers.polyaxon_logger import *

            @trainer.on(Events.ITERATION_COMPLETED(every=500))
            def evaluate(engine):
                evaluator.run(validation_set, max_epochs=1)

            plx_logger = PolyaxonLogger()

            def global_step_transform(*args, **kwargs):
                return trainer.state.iteration

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # every 500 iterations. Since evaluator engine does not have access to the training iteration, we
            # provide a global_step_transform to return the trainer.state.iteration for the global_step, each time
            # evaluator metrics are plotted on Polyaxon.

            plx_logger.attach(evaluator,
                              log_handler=OutputHandler(tag="validation",
                                                        metrics=["nll", "accuracy"],
                                                        global_step_transform=global_step_transform),
                              event_name=Events.EPOCH_COMPLETED)

    Args:
        tag (str): common title for all produced plots. For example, 'training'
        metric_names (list of str, optional): list of metric names to plot or a string "all" to plot all available
            metrics.
        output_transform (callable, optional): output transform function to prepare `engine.state.output` as a number.
            For example, `output_transform = lambda output: output`
            This function can also return a dictionary, e.g `{'loss': loss1, 'another_loss': loss2}` to label the plot
            with corresponding keys.
        another_engine (Engine): Deprecated (see :attr:`global_step_transform`). Another engine to use to provide the
            value of event. Typically, user can provide
            the trainer if this handler is attached to an evaluator and thus it logs proper trainer's
            epoch/iteration value.
        global_step_transform (callable, optional): global step transform function to output a desired global step.
            Input of the function is `(engine, event_name)`. Output of function should be an integer.
            Default is None, global_step based on attached engine. If provided,
            uses function output as global_step. To setup global step from another engine, please use
            :meth:`~ignite.contrib.handlers.polyaxon_logger.global_step_from_engine`.

    Note:

        Example of `global_step_transform`:

        .. code-block:: python

            def global_step_transform(engine, event_name):
                return engine.state.get_event_attrib_value(event_name)

    """

    def __init__(self, tag, metric_names=None, output_transform=None, another_engine=None, global_step_transform=None):
        super(OutputHandler, self).__init__(tag, metric_names, output_transform, another_engine, global_step_transform)

    def __call__(self, engine, logger, event_name):

        if not isinstance(logger, PolyaxonLogger):
            raise RuntimeError("Handler 'OutputHandler' works only with PolyaxonLogger")

        metrics = self._setup_output_metrics(engine)

        global_step = self.global_step_transform(engine, event_name)

        if not isinstance(global_step, int):
            raise TypeError("global_step must be int, got {}."
                            " Please check the output of global_step_transform.".format(type(global_step)))

        rendered_metrics = {"step": global_step}
        for key, value in metrics.items():
            if isinstance(value, numbers.Number):
                rendered_metrics["{}/{}".format(self.tag, key)] = value
            elif isinstance(value, torch.Tensor) and value.ndimension() == 0:
                rendered_metrics["{}/{}".format(self.tag, key)] = value.item()
            elif isinstance(value, torch.Tensor) and value.ndimension() == 1:
                for i, v in enumerate(value):
                    rendered_metrics["{}/{}/{}".format(self.tag, key, i)] = v.item()
            else:
                warnings.warn("PolyaxonLogger output_handler can not log "
                              "metrics value type {}".format(type(value)))

        logger.log_metrics(**rendered_metrics)


class OptimizerParamsHandler(BaseOptimizerParamsHandler):
    """Helper handler to log optimizer parameters

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.polyaxon_logger import *

            # Create a logger
            plx_logger = PolyaxonLogger()

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            plx_logger.attach(trainer,
                              log_handler=OptimizerParamsHandler(optimizer),
                              event_name=Events.ITERATION_STARTED)

    Args:
        optimizer (torch.optim.Optimizer): torch optimizer which parameters to log
        param_name (str): parameter name
        tag (str, optional): common title for all produced plots. For example, 'generator'
    """

    def __init__(self, optimizer, param_name="lr", tag=None):
        super(OptimizerParamsHandler, self).__init__(optimizer, param_name, tag)

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, PolyaxonLogger):
            raise RuntimeError("Handler 'OptimizerParamsHandler' works only with PolyaxonLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        params = {"{}{}/group_{}".format(tag_prefix, self.param_name, i): float(param_group[self.param_name])
                  for i, param_group in enumerate(self.optimizer.param_groups)}
        params['step'] = global_step
        logger.log_metrics(**params)


class PolyaxonLogger(BaseLogger):
    """
    `Polyaxon <https://polyaxon.com/>`_ tracking client handler to log parameters and metrics during the training
    and validation.

    This class requires `polyaxon-client <https://github.com/polyaxon/polyaxon-client/>`_ package to be installed:

    .. code-block:: bash

        pip install polyaxon-client


    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.polyaxon_logger import *

            # Create a logger
            plx_logger = PolyaxonLogger()

            # Log experiment parameters:
            plx_logger.log_params(**{
                "seed": seed,
                "batch_size": batch_size,
                "model": model.__class__.__name__,

                "pytorch version": torch.__version__,
                "ignite version": ignite.__version__,
                "cuda version": torch.version.cuda,
                "device name": torch.cuda.get_device_name(0)
            })

            # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
            # We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer` instead of `train_evaluator`.
            plx_logger.attach(train_evaluator,
                              log_handler=OutputHandler(tag="training",
                                                        metric_names=["nll", "accuracy"],
                                                        global_step_transform=global_step_from_engine(trainer)),
                              event_name=Events.EPOCH_COMPLETED)

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
            # `trainer` instead of `evaluator`.
            plx_logger.attach(evaluator,
                              log_handler=OutputHandler(tag="validation",
                                                        metric_names=["nll", "accuracy"],
                                                        global_step_transform=global_step_from_engine(trainer)),
                              event_name=Events.EPOCH_COMPLETED)
    """

    def __init__(self):
        try:
            from polyaxon_client.tracking import Experiment
        except ImportError:
            raise RuntimeError("This contrib module requires polyaxon-client to be installed. "
                               "Please install it with command: \n pip install polyaxon-client")

        self.experiment = Experiment()

    def __getattr__(self, attr):
        def wrapper(*args, **kwargs):
            return getattr(self.experiment, attr)(*args, **kwargs)

        return wrapper
