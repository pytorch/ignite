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

        if not isinstance(logger, PolyaxonLogger):
            raise RuntimeError("Handler 'OutputHandler' works only with PolyaxonLogger")

        metrics = self._setup_output_metrics(engine)

        state = engine.state if self.another_engine is None else self.another_engine.state
        global_step = state.get_event_attrib_value(event_name)

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
