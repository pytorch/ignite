import os
import numbers

import warnings
import torch

from ignite.contrib.handlers.base_logger import BaseLogger, BaseOptimizerParamsHandler, BaseOutputHandler, \
    BaseWeightsScalarHandler


__all__ = ['VisdomLogger', 'OptimizerParamsHandler', 'OutputHandler',
           'WeightsScalarHandler', 'GradsScalarHandler']


class _BaseVisDrawer(object):

    def __init__(self, show_legend=False):
        self.windows = {}
        self.show_legend = show_legend

    def add_scalar(self, logger, k, v, event_name, global_step):
        """
        Helper method to log a scalar with VisdomLogger.

        Args:
            logger (VisdomLogger): visdom logger
            k (str): scalar name which is used to set window title and y-axis label
            v (int or float): scalar value, y-axis value
            event_name: Event name which is used to setup x-axis label. Valid events are from
                :class:`~ignite.engine.Events` or any `event_name` added by
                :meth:`~ignite.engine.Engine.register_events`.
            global_step (int): global step, x-axis value

        """
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

        kwargs = {
            "X": [global_step, ],
            "Y": [v, ],
            "env": logger.vis.env,
            "win": self.windows[k]['win'],
            "update": update,
            "opts": self.windows[k]['opts'],
            "name": k
        }

        future = logger.executor.submit(logger.vis.line, **kwargs)
        if self.windows[k]['win'] is None:
            self.windows[k]['win'] = future.result()


class OutputHandler(BaseOutputHandler, _BaseVisDrawer):
    """Helper handler to log engine's output and/or metrics

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.visdom_logger import *

            # Create a logger
            vd_logger = VisdomLogger()

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `another_engine=trainer` to take the epoch of the `trainer`
            vd_logger.attach(evaluator,
                             log_handler=OutputHandler(tag="validation",
                                                       metric_names=["nll", "accuracy"],
                                                       another_engine=trainer),
                             event_name=Events.EPOCH_COMPLETED)

        Example with CustomPeriodicEvent, where model is evaluated every 500 iterations:

        .. code-block:: python

            from ignite.contrib.handlers import CustomPeriodicEvent

            cpe = CustomPeriodicEvent(n_iterations=500)
            cpe.attach(trainer)

            @trainer.on(cpe.Events.ITERATIONS_500_COMPLETED)
            def evaluate(engine):
                evaluator.run(validation_set, max_epochs=1)

            from ignite.contrib.handlers.visdom_logger import *

            vd_logger = VisdomLogger()

            def global_step_transform(*args, **kwargs):
                return trainer.state.iteration

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # every 500 iterations. Since evaluator engine does not have CustomPeriodicEvent attached to it, we
            # provide a global_step_transform to return the trainer.state.iteration for the global_step, each time
            # evaluator metrics are plotted on Visdom.


            vd_logger.attach(evaluator,
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
            This function can also return a dictionary, e.g `{'loss': loss1, `another_loss`: loss2}` to label the plot
            with corresponding keys.
        another_engine (Engine): another engine to use to provide the value of event. Typically, user can provide
            the trainer if this handler is attached to an evaluator and thus it logs proper trainer's
            epoch/iteration value.
        global_step_transform (callable, optional): global step transform function to output a desired global step.
            Output of function should be an integer. Default is None, global_step based on attached engine. If provided,
            uses function output as global_step.
        show_legend (bool, optional): flag to show legend in the window
    """

    def __init__(self, tag, metric_names=None, output_transform=None, another_engine=None, global_step_transform=None,
                 show_legend=False):
        super(OutputHandler, self).__init__(tag, metric_names, output_transform, another_engine, global_step_transform)
        _BaseVisDrawer.__init__(self, show_legend=show_legend)

    def __call__(self, engine, logger, event_name):

        if not isinstance(logger, VisdomLogger):
            raise RuntimeError("Handler 'OutputHandler' works only with VisdomLogger")

        metrics = self._setup_output_metrics(engine)

        engine = engine if self.another_engine is None else self.another_engine
        global_step = self.global_step_transform(engine, event_name)
        if not isinstance(global_step, int):
            raise TypeError("global_step must be int, got {}."
                            " Please check the output of global_step_transform.".format(type(global_step)))

        for key, value in metrics.items():

            values = []
            keys = []
            if isinstance(value, numbers.Number) or \
                    isinstance(value, torch.Tensor) and value.ndimension() == 0:
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
                self.add_scalar(logger, k, v, event_name, global_step)

        logger._save()


class OptimizerParamsHandler(BaseOptimizerParamsHandler, _BaseVisDrawer):
    """Helper handler to log optimizer parameters

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.visdom_logger import *

            # Create a logger
            vb_logger = VisdomLogger()

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            vb_logger.attach(trainer,
                             log_handler=OptimizerParamsHandler(optimizer),
                             event_name=Events.ITERATION_STARTED)

    Args:
        optimizer (torch.optim.Optimizer): torch optimizer which parameters to log
        param_name (str): parameter name
        tag (str, optional): common title for all produced plots. For example, 'generator'
        show_legend (bool, optional): flag to show legend in the window
    """

    def __init__(self, optimizer, param_name="lr", tag=None, show_legend=False):
        super(OptimizerParamsHandler, self).__init__(optimizer, param_name, tag)
        _BaseVisDrawer.__init__(self, show_legend=show_legend)

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, VisdomLogger):
            raise RuntimeError("Handler 'OptimizerParamsHandler' works only with VisdomLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        params = {"{}{}/group_{}".format(tag_prefix, self.param_name, i): float(param_group[self.param_name])
                  for i, param_group in enumerate(self.optimizer.param_groups)}

        for k, v in params.items():
            self.add_scalar(logger, k, v, event_name, global_step)

        logger._save()


class WeightsScalarHandler(BaseWeightsScalarHandler, _BaseVisDrawer):
    """Helper handler to log model's weights as scalars.
    Handler iterates over named parameters of the model, applies reduction function to each parameter
    produce a scalar and then logs the scalar.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.visdom_logger import *

            # Create a logger
            vd_logger = VisdomLogger()

            # Attach the logger to the trainer to log model's weights norm after each iteration
            vd_logger.attach(trainer,
                             log_handler=WeightsScalarHandler(model, reduction=torch.norm),
                             event_name=Events.ITERATION_COMPLETED)

    Args:
        model (torch.nn.Module): model to log weights
        reduction (callable): function to reduce parameters into scalar
        tag (str, optional): common title for all produced plots. For example, 'generator'
        show_legend (bool, optional): flag to show legend in the window
    """

    def __init__(self, model, reduction=torch.norm, tag=None, show_legend=False):
        super(WeightsScalarHandler, self).__init__(model, reduction, tag=tag)
        _BaseVisDrawer.__init__(self, show_legend=show_legend)

    def __call__(self, engine, logger, event_name):

        if not isinstance(logger, VisdomLogger):
            raise RuntimeError("Handler 'WeightsScalarHandler' works only with VisdomLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        for name, p in self.model.named_parameters():
            name = name.replace('.', '/')
            k = "{}weights_{}/{}".format(tag_prefix, self.reduction.__name__, name)
            v = float(self.reduction(p.data))
            self.add_scalar(logger, k, v, event_name, global_step)

        logger._save()


class GradsScalarHandler(BaseWeightsScalarHandler, _BaseVisDrawer):
    """Helper handler to log model's gradients as scalars.
    Handler iterates over the gradients of named parameters of the model, applies reduction function to each parameter
    produce a scalar and then logs the scalar.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.visdom_logger import *

            # Create a logger
            vd_logger = VisdomLogger()

            # Attach the logger to the trainer to log model's weights norm after each iteration
            vd_logger.attach(trainer,
                             log_handler=GradsScalarHandler(model, reduction=torch.norm),
                             event_name=Events.ITERATION_COMPLETED)

    Args:
        model (torch.nn.Module): model to log weights
        reduction (callable): function to reduce parameters into scalar
        tag (str, optional): common title for all produced plots. For example, 'generator'
        show_legend (bool, optional): flag to show legend in the window

    """

    def __init__(self, model, reduction=torch.norm, tag=None, show_legend=False):
        super(GradsScalarHandler, self).__init__(model, reduction, tag)
        _BaseVisDrawer.__init__(self, show_legend=show_legend)

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, VisdomLogger):
            raise RuntimeError("Handler 'GradsScalarHandler' works only with VisdomLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        for name, p in self.model.named_parameters():
            name = name.replace('.', '/')
            k = "{}grads_{}/{}".format(tag_prefix, self.reduction.__name__, name)
            v = float(self.reduction(p.grad))
            self.add_scalar(logger, k, v, event_name, global_step)

        logger._save()


class VisdomLogger(BaseLogger):
    """
    VisdomLogger handler to log metrics, model/optimizer parameters, gradients during the training and validation.

    This class requires `visdom <https://github.com/facebookresearch/visdom/>`_ package to be installed:

    .. code-block:: bash

        pip install git+https://github.com/facebookresearch/visdom.git

    Args:
        server (str, optional): visdom server URL. It can be also specified by environment variable `VISDOM_SERVER_URL`
        port (int, optional): visdom server's port. It can be also specified by environment variable `VISDOM_PORT`
        num_workers (int, optional): number of workers to use in `concurrent.futures.ThreadPoolExecutor` to post data to
            visdom server. Default, `num_workers=1`. If `num_workers=0` and logger uses the main thread. If using
            Python 2.7 and `num_workers>0` the package `futures` should be installed: `pip install futures`
        **kwargs: kwargs to pass into
            `visdom.Visdom <https://github.com/facebookresearch/visdom#visdom-arguments-python-only>`_.

    Note:
        We can also specify username/password using environment variables: VISDOM_USERNAME, VISDOM_PASSWORD


    .. warning::

        Frequent logging, e.g. when logger is attached to `Events.ITERATION_COMPLETED`, can slow down the run if the
        main thread is used to send the data to visdom server (`num_workers=0`). To avoid this situation we can either
        log less frequently or set `num_workers=1`.


    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.visdom_logger import *

            # Create a logger
            vd_logger = VisdomLogger()

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

            # Attach the logger to the trainer to log model's gradients norm after each iteration
            vd_logger.attach(trainer,
                             log_handler=grads_scalar_handler(model),
                             event_name=Events.ITERATION_COMPLETED)

            # We need to close the logger with we are done
            vd_logger.close()

        It is also possible to use the logger as context manager:

        .. code-block:: python

            from ignite.contrib.handlers.visdom_logger import *

            with VisdomLogger() as vd_logger:

                trainer = Engine(update_fn)
                # Attach the logger to the trainer to log training loss at each iteration
                vd_logger.attach(trainer,
                                 log_handler=OutputHandler(tag="training",
                                                           output_transform=lambda loss: {'loss': loss}),
                                 event_name=Events.ITERATION_COMPLETED)



    """

    def __init__(self, server=None, port=None, num_workers=1, **kwargs):
        try:
            import visdom
        except ImportError:
            raise RuntimeError("This contrib module requires visdom package. "
                               "Please install it with command:\n"
                               "pip install git+https://github.com/facebookresearch/visdom.git")

        if num_workers > 0:
            # If visdom is installed, one of its dependencies `tornado`
            # requires also `futures` to be installed.
            # Let's check anyway if we can import it.
            try:
                import concurrent.futures
            except ImportError:
                raise RuntimeError("This contrib module requires concurrent.futures module"
                                   "Please install it with command:\n"
                                   "pip install futures")

        if server is None:
            server = os.environ.get("VISDOM_SERVER_URL", 'localhost')

        if port is None:
            port = int(os.environ.get("VISDOM_PORT", 8097))

        if "username" not in kwargs:
            username = os.environ.get("VISDOM_USERNAME", None)
            kwargs["username"] = username

        if "password" not in kwargs:
            password = os.environ.get("VISDOM_PASSWORD", None)
            kwargs["password"] = password

        self.vis = visdom.Visdom(
            server=server,
            port=port,
            **kwargs
        )

        if not self.vis.check_connection():
            raise RuntimeError("Failed to connect to Visdom server at {}. "
                               "Did you run python -m visdom.server ?".format(server))

        self.executor = _DummyExecutor()
        if num_workers > 0:
            from concurrent.futures import ThreadPoolExecutor
            self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def _save(self):
        self.vis.save([self.vis.env])

    def close(self):
        self.vis = None
        self.executor.shutdown()


class _DummyExecutor:

    class _DummyFuture:

        def __init__(self, result):
            self._output = result

        def result(self):
            return self._output

    def __init__(self, *args, **kwargs):
        pass

    def submit(self, fn, **kwargs):
        return _DummyExecutor._DummyFuture(fn(**kwargs))

    def shutdown(self, *args, **kwargs):
        pass
