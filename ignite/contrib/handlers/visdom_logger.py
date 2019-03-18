import os
import numbers

import warnings
import torch

from ignite.contrib.handlers.base_logger import BaseLogger, BaseOptimizerParamsHandler, BaseOutputHandler, \
    BaseWeightsScalarHandler, BaseWeightsHistHandler


__all__ = ['VisdomLogger', 'OutputHandler']


class OutputHandler(BaseOutputHandler):
    """Helper handler to log engine's output and/or metrics

    Examples:

        ..code-block:: python

            from ignite.contrib.handlers.visdom_logger import *

            # Create a logger
            vd_logger = VisdomLogger(log_dir="experiments/vd_logs")

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `another_engine=trainer` to take the epoch of the `trainer`
            vd_logger.attach(evaluator,
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
        env (): visdom's environment passed to trace lines
        show_legend (bool, optional): flag to show legend in the window
    """

    def __init__(self, tag, metric_names=None, output_transform=None, another_engine=None,
                 env=None, show_legend=False):
        super(OutputHandler, self).__init__(tag, metric_names, output_transform, another_engine)

        self.windows = {}
        self.env = env
        self.show_legend = show_legend

    def __call__(self, engine, logger, event_name):

        if not isinstance(logger, VisdomLogger):
            raise RuntimeError("Handler 'OutputHandler' works only with VisdomLogger")

        metrics = self._setup_output_metrics(engine)

        state = engine.state if self.another_engine is None else self.another_engine.state
        global_step = state.get_event_attrib_value(event_name)

        for key, value in metrics.items():

            values = []
            keys = []
            if isinstance(value, numbers.Number):
                values.append(value)
                keys.append(key)
            elif isinstance(value, torch.Tensor) and value.ndimension() == 1:
                values = value
                keys = ["{}/{}".format(key, i) for i in range(len(value))]
            else:
                warnings.warn("VisdomLogger output_handler can not log "
                              "metrics value type {}".format(type(value)))

            for k, v in zip(keys, values):
                k = "{}/{}".format(self.tag, k)
                if k not in self.windows:
                    self.windows[k] = {
                        'win': None,
                        'opts': {
                            'title': k,
                            'xlabel': str(event_name),
                            'ylabel': k,
                            'showlegend': self.show_legend
                        }
                    }

                update = None if self.windows[k]['win'] is None else 'append'
                ret = logger.vis.line(
                    X=[global_step, ],
                    Y=[v, ],
                    env=self.env,
                    win=self.windows[k]['win'],
                    update=update,
                    opts=self.windows[k]['opts'],
                    name=k,
                )
                if self.windows[k]['win'] is None:
                    self.windows[k]['win'] = ret


class OptimizerParamsHandler(BaseOptimizerParamsHandler):

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


class VisdomLogger(BaseLogger):
    """
    VisdomLogger handler to log metrics, model/optimizer parameters, gradients during the training and validation.

    Args:
        log_dir (str): path to the directory where to log.

    Examples:

        ..code-block:: python

            from ignite.contrib.handlers.visdom_logger import *

            # Create a logger
            vd_logger = VisdomLogger(log_dir="experiments/vd_logs")

            # Attach the logger to the trainer to log training loss at each iteration
            vd_logger.attach(trainer,
                             log_handler=OutputHandler(tag="training", output_transform=lambda loss: {'loss': loss}),
                             event_name=Events.ITERATION_COMPLETED)

            # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
            # We setup `another_engine=trainer` to take the epoch of the `trainer` instead of `train_evaluator`.
            vd_logger.attach(train_evaluator,
                             log_handler=OutputHandler(tag="training",
                                                       metric_names=["nll", "accuracy"],
                                                       another_engine=trainer),
                             event_name=Events.EPOCH_COMPLETED)

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `another_engine=trainer` to take the epoch of the `trainer` instead of `evaluator`.
            vd_logger.attach(evaluator,
                             log_handler=OutputHandler(tag="validation",
                                                       metric_names=["nll", "accuracy"],
                                                       another_engine=trainer),
                             event_name=Events.EPOCH_COMPLETED)

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            vd_logger.attach(trainer,
                             log_handler=optimizer_params_handler(optimizer),
                             event_name=Events.ITERATION_COMPLETED)

            # Attach the logger to the trainer to log model's weights norm after each iteration
            vd_logger.attach(trainer,
                             log_handler=weights_scalar_handler(model),
                             event_name=Events.ITERATION_COMPLETED)

            # Attach the logger to the trainer to log model's weights as a histogram after each epoch
            vd_logger.attach(trainer,
                             log_handler=weights_hist_handler(model),
                             event_name=Events.EPOCH_COMPLETED)

            # Attach the logger to the trainer to log model's gradients norm after each iteration
            vd_logger.attach(trainer,
                             log_handler=grads_scalar_handler(model),
                             event_name=Events.ITERATION_COMPLETED)

            # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
            vd_logger.attach(trainer,
                             log_handler=grads_hist_handler(model),
                             event_name=Events.EPOCH_COMPLETED)

    """

    def __init__(self, log_dir, server=None, port=None, **kwargs):
        try:
            import visdom
        except ImportError:
            raise RuntimeError("This contrib module requires visdom."
                               "Please install it with command:\n"
                               "pip install git+https://github.com/facebookresearch/visdom.git")
        if server is None:
            server = os.environ.get("VISDOM_SERVER_URL", 'http://localhost')

        if port is None:
            port = os.environ.get("VISDOM_PORT", 8097)

        username = os.environ.get("VISDOM_USERNAME", None)
        password = os.environ.get("VISDOM_PASSWORD", None)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        self.vis = visdom.Visdom(
            server=server,
            port=port,
            log_to_filename=os.path.join(log_dir, "logging_{}.visdom".format(id(self))),
            username=username,
            password=password,
            **kwargs
        )

        if not self.vis.check_connection():
            raise RuntimeError("Failed to connect to Visdom server at {}. "
                               "Did you run python -m visdom.server ?".format(server))

    def _close(self):
        self.vis = None
