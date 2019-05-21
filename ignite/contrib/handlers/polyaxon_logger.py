import numbers

import warnings
import torch

from ignite.contrib.handlers.base_logger import BaseLogger, BaseOutputHandler


__all__ = ['PolyaxonLogger', 'OutputHandler']


class OutputHandler(BaseOutputHandler):
    """Helper handler to log engine's output and/or metrics.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.polyaxon_logger import *

            # Create a logger
            plx_logger = PolyaxonLogger()

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `another_engine=trainer` to take the epoch of the `trainer`
            plx_logger.attach(evaluator,
                              log_handler=OutputHandler(tag="validation",
                                                        metric_names=["nll", "accuracy"],
                                                        another_engine=trainer),
                              event_name=Events.EPOCH_COMPLETED)

        Example with CustomPeriodicEvent, where model is evaluated every 500 iterations:
        ..code-block:: python

            from ignite.contrib.handlers import CustomPeriodicEvent

            cpe = CustomPeriodicEvent(n_iterations=500)
            cpe.attach(trainer)

            @trainer.on(cpe.Events.ITERATIONS_500_COMPLETED)
            def evaluate(engine):
                evaluator.run(validation_set, max_epochs=1)

            from ignite.contrib.handlers.polyaxon_logger import *

            plx_logger = PolyaxonLogger()

            def global_step_transform(*args, **kwargs):
                return trainer.state.iteration

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # every 500 iterations. Since evaluator engine does not have CustomPeriodicEvent attached to it, we
            # provide a global_step_transform to return the trainer.state.iteration for the global_step, each time
            # evaluator metrics are plotted on Polyaxon.


            plx_logger.attach(evaluator,
                              log_handler=OutputHandler(tag="validation",
                                                        metrics=["nll", "accuracy"],
                                                        global_step_transform=global_step_transform),
                              event_name=Events.EPOCH_COMPLETED)

    Args:
        tag (str): common title for all produced plots. For example, 'training'
        metric_names (list of str, optional): list of metric names to plot.
        output_transform (callable, optional): output transform function to prepare `engine.state.output` as a number.
            For example, `output_transform = lambda output: output`
            This function can also return a dictionary, e.g `{'loss': loss1, `another_loss`: loss2}` to label the plot
            with corresponding keys.
        global_step_transform (callable, optional): global step transform function to output a desired global step.
            Output of function should be an integer. Default is None, global_step based on attached engine. If provided,
            uses function output as global_step.
    """
    def __init__(self, tag, metric_names=None, output_transform=None, global_step_transform=None):
        super(OutputHandler, self).__init__(tag, metric_names, output_transform, global_step_transform)

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
            # We setup `another_engine=trainer` to take the epoch of the `trainer` instead of `train_evaluator`.
            plx_logger.attach(train_evaluator,
                              log_handler=OutputHandler(tag="training",
                                                        metric_names=["nll", "accuracy"],
                                                        another_engine=trainer),
                              event_name=Events.EPOCH_COMPLETED)

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `another_engine=trainer` to take the epoch of the `trainer`
            plx_logger.attach(evaluator,
                              log_handler=OutputHandler(tag="validation",
                                                        metric_names=["nll", "accuracy"],
                                                        another_engine=trainer),
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
