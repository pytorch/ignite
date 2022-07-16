"""Visdom logger and its helper handlers."""
import os
from typing import Any, Callable, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ignite.contrib.handlers.base_logger import (
    BaseLogger,
    BaseOptimizerParamsHandler,
    BaseOutputHandler,
    BaseWeightsScalarHandler,
)
from ignite.engine import Engine, Events
from ignite.handlers import global_step_from_engine

__all__ = [
    "VisdomLogger",
    "OptimizerParamsHandler",
    "OutputHandler",
    "WeightsScalarHandler",
    "GradsScalarHandler",
    "global_step_from_engine",
]


class VisdomLogger(BaseLogger):
    """
    VisdomLogger handler to log metrics, model/optimizer parameters, gradients during the training and validation.

    This class requires `visdom <https://github.com/fossasia/visdom/>`_ package to be installed:

    .. code-block:: bash


        pip install git+https://github.com/fossasia/visdom.git

    Args:
        server: visdom server URL. It can be also specified by environment variable `VISDOM_SERVER_URL`
        port: visdom server's port. It can be also specified by environment variable `VISDOM_PORT`
        num_workers: number of workers to use in `concurrent.futures.ThreadPoolExecutor` to post data to
            visdom server. Default, `num_workers=1`. If `num_workers=0` and logger uses the main thread. If using
            Python 2.7 and `num_workers>0` the package `futures` should be installed: `pip install futures`
        kwargs: kwargs to pass into
            `visdom.Visdom <https://github.com/fossasia/visdom#user-content-visdom-arguments-python-only>`_.

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
            vd_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="training",
                output_transform=lambda loss: {"loss": loss}
            )

            # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
            # We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer` instead of `train_evaluator`.
            vd_logger.attach_output_handler(
                train_evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="training",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer),
            )

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
            # `trainer` instead of `evaluator`.
            vd_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer)),
            )

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            vd_logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                optimizer=optimizer,
                param_name='lr'  # optional
            )

            # Attach the logger to the trainer to log model's weights norm after each iteration
            vd_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(model)
            )

            # Attach the logger to the trainer to log model's gradients norm after each iteration
            vd_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsScalarHandler(model)
            )

            # We need to close the logger with we are done
            vd_logger.close()

        It is also possible to use the logger as context manager:

        .. code-block:: python

            from ignite.contrib.handlers.visdom_logger import *

            with VisdomLogger() as vd_logger:

                trainer = Engine(update_fn)
                # Attach the logger to the trainer to log training loss at each iteration
                vd_logger.attach_output_handler(
                    trainer,
                    event_name=Events.ITERATION_COMPLETED,
                    tag="training",
                    output_transform=lambda loss: {"loss": loss}
                )

    ..  versionchanged:: 0.5.0
        accepts an optional list of `state_attributes`
    """

    def __init__(
        self,
        server: Optional[str] = None,
        port: Optional[int] = None,
        num_workers: int = 1,
        raise_exceptions: bool = True,
        **kwargs: Any,
    ):
        try:
            import visdom
        except ImportError:
            raise RuntimeError(
                "This contrib module requires visdom package. "
                "Please install it with command:\n"
                "pip install git+https://github.com/fossasia/visdom.git"
            )

        if num_workers > 0:
            # If visdom is installed, one of its dependencies `tornado`
            # requires also `futures` to be installed.
            # Let's check anyway if we can import it.
            try:
                from concurrent.futures import ThreadPoolExecutor
            except ImportError:
                raise RuntimeError(
                    "This contrib module requires concurrent.futures module"
                    "Please install it with command:\n"
                    "pip install futures"
                )

        if server is None:
            server = cast(str, os.environ.get("VISDOM_SERVER_URL", "localhost"))

        if port is None:
            port = int(os.environ.get("VISDOM_PORT", 8097))

        if "username" not in kwargs:
            username = os.environ.get("VISDOM_USERNAME", None)
            kwargs["username"] = username

        if "password" not in kwargs:
            password = os.environ.get("VISDOM_PASSWORD", None)
            kwargs["password"] = password

        self.vis = visdom.Visdom(server=server, port=port, raise_exceptions=raise_exceptions, **kwargs)

        if not self.vis.offline and not self.vis.check_connection():  # type: ignore[attr-defined]
            raise RuntimeError(f"Failed to connect to Visdom server at {server}. Did you run python -m visdom.server ?")

        self.executor = _DummyExecutor()  # type: Union[_DummyExecutor, "ThreadPoolExecutor"]
        if num_workers > 0:
            self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def _save(self) -> None:
        self.vis.save([self.vis.env])  # type: ignore[attr-defined]

    def close(self) -> None:
        self.executor.shutdown()
        self.vis.close()

    def _create_output_handler(self, *args: Any, **kwargs: Any) -> "OutputHandler":
        return OutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args: Any, **kwargs: Any) -> "OptimizerParamsHandler":
        return OptimizerParamsHandler(*args, **kwargs)


class _BaseVisDrawer:
    def __init__(self, show_legend: bool = False):
        self.windows = {}  # type: Dict[str, Any]
        self.show_legend = show_legend

    def add_scalar(
        self, logger: VisdomLogger, k: str, v: Union[str, float, torch.Tensor], event_name: Any, global_step: int
    ) -> None:
        """
        Helper method to log a scalar with VisdomLogger.

        Args:
            logger: visdom logger
            k: scalar name which is used to set window title and y-axis label
            v: scalar value, y-axis value
            event_name: Event name which is used to setup x-axis label. Valid events are from
                :class:`~ignite.engine.events.Events` or any `event_name` added by
                :meth:`~ignite.engine.engine.Engine.register_events`.
            global_step: global step, x-axis value

        """
        if k not in self.windows:
            self.windows[k] = {
                "win": None,
                "opts": {"title": k, "xlabel": str(event_name), "ylabel": k, "showlegend": self.show_legend},
            }

        update = None if self.windows[k]["win"] is None else "append"

        kwargs = {
            "X": [global_step],
            "Y": [v],
            "env": logger.vis.env,  # type: ignore[attr-defined]
            "win": self.windows[k]["win"],
            "update": update,
            "opts": self.windows[k]["opts"],
            "name": k,
        }

        future = logger.executor.submit(logger.vis.line, **kwargs)
        if self.windows[k]["win"] is None:
            self.windows[k]["win"] = future.result()


class OutputHandler(BaseOutputHandler, _BaseVisDrawer):
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
            :meth:`~ignite.contrib.handlers.visdom_logger.global_step_from_engine`.
        show_legend: flag to show legend in the window
        state_attributes: list of attributes of the ``trainer.state`` to plot.

    Examples:
        .. code-block:: python

            from ignite.contrib.handlers.visdom_logger import *

            # Create a logger
            vd_logger = VisdomLogger()

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer`:
            vd_logger.attach(
                evaluator,
                log_handler=OutputHandler(
                    tag="validation",
                    metric_names=["nll", "accuracy"],
                    global_step_transform=global_step_from_engine(trainer)
                ),
                event_name=Events.EPOCH_COMPLETED
            )
            # or equivalently
            vd_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=["nll", "accuracy"],
                global_step_transform=global_step_from_engine(trainer)
            )

        Another example, where model is evaluated every 500 iterations:

        .. code-block:: python

            from ignite.contrib.handlers.visdom_logger import *

            @trainer.on(Events.ITERATION_COMPLETED(every=500))
            def evaluate(engine):
                evaluator.run(validation_set, max_epochs=1)

            vd_logger = VisdomLogger()

            def global_step_transform(*args, **kwargs):
                return trainer.state.iteration

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # every 500 iterations. Since evaluator engine does not have access to the training iteration, we
            # provide a global_step_transform to return the trainer.state.iteration for the global_step, each time
            # evaluator metrics are plotted on Visdom.

            vd_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metrics=["nll", "accuracy"],
                global_step_transform=global_step_transform
            )

        Another example where the State Attributes ``trainer.state.alpha`` and ``trainer.state.beta``
        are also logged along with the NLL and Accuracy after each iteration:

        .. code-block:: python

            vd_logger.attach(
                trainer,
                log_handler=OutputHandler(
                    tag="training",
                    metric_names=["nll", "accuracy"],
                    state_attributes=["alpha", "beta"],
                ),
                event_name=Events.ITERATION_COMPLETED
            )

        Example of `global_step_transform`:

        .. code-block:: python

            def global_step_transform(engine, event_name):
                return engine.state.get_event_attrib_value(event_name)
    """

    def __init__(
        self,
        tag: str,
        metric_names: Optional[str] = None,
        output_transform: Optional[Callable] = None,
        global_step_transform: Optional[Callable] = None,
        show_legend: bool = False,
        state_attributes: Optional[List[str]] = None,
    ):
        super(OutputHandler, self).__init__(
            tag, metric_names, output_transform, global_step_transform, state_attributes
        )
        _BaseVisDrawer.__init__(self, show_legend=show_legend)

    def __call__(self, engine: Engine, logger: VisdomLogger, event_name: Union[str, Events]) -> None:

        if not isinstance(logger, VisdomLogger):
            raise RuntimeError("Handler 'OutputHandler' works only with VisdomLogger")

        metrics = self._setup_output_metrics_state_attrs(engine, key_tuple=False)

        global_step = self.global_step_transform(engine, event_name)  # type: ignore[misc]

        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}."
                " Please check the output of global_step_transform."
            )

        for key, value in metrics.items():
            self.add_scalar(logger, key, value, event_name, global_step)

        logger._save()


class OptimizerParamsHandler(BaseOptimizerParamsHandler, _BaseVisDrawer):
    """Helper handler to log optimizer parameters

    Args:
        optimizer: torch optimizer or any object with attribute ``param_groups``
            as a sequence.
        param_name: parameter name
        tag: common title for all produced plots. For example, "generator"
        show_legend: flag to show legend in the window

    Examples:
        .. code-block:: python

            from ignite.contrib.handlers.visdom_logger import *

            # Create a logger
            vb_logger = VisdomLogger()

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            vd_logger.attach(
                trainer,
                log_handler=OptimizerParamsHandler(optimizer),
                event_name=Events.ITERATION_STARTED
            )
            # or equivalently
            vd_logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                optimizer=optimizer
            )
    """

    def __init__(
        self, optimizer: Optimizer, param_name: str = "lr", tag: Optional[str] = None, show_legend: bool = False
    ):
        super(OptimizerParamsHandler, self).__init__(optimizer, param_name, tag)
        _BaseVisDrawer.__init__(self, show_legend=show_legend)

    def __call__(self, engine: Engine, logger: VisdomLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, VisdomLogger):
            raise RuntimeError("Handler OptimizerParamsHandler works only with VisdomLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        params = {
            f"{tag_prefix}{self.param_name}/group_{i}": float(param_group[self.param_name])
            for i, param_group in enumerate(self.optimizer.param_groups)
        }

        for k, v in params.items():
            self.add_scalar(logger, k, v, event_name, global_step)

        logger._save()


class WeightsScalarHandler(BaseWeightsScalarHandler, _BaseVisDrawer):
    """Helper handler to log model's weights as scalars.
    Handler iterates over named parameters of the model, applies reduction function to each parameter
    produce a scalar and then logs the scalar.

    Args:
        model: model to log weights
        reduction: function to reduce parameters into scalar
        tag: common title for all produced plots. For example, "generator"
        show_legend: flag to show legend in the window

    Examples:
        .. code-block:: python

            from ignite.contrib.handlers.visdom_logger import *

            # Create a logger
            vd_logger = VisdomLogger()

            # Attach the logger to the trainer to log model's weights norm after each iteration
            vd_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=WeightsScalarHandler(model, reduction=torch.norm)
            )
    """

    def __init__(
        self, model: nn.Module, reduction: Callable = torch.norm, tag: Optional[str] = None, show_legend: bool = False
    ):
        super(WeightsScalarHandler, self).__init__(model, reduction, tag=tag)
        _BaseVisDrawer.__init__(self, show_legend=show_legend)

    def __call__(self, engine: Engine, logger: VisdomLogger, event_name: Union[str, Events]) -> None:

        if not isinstance(logger, VisdomLogger):
            raise RuntimeError("Handler 'WeightsScalarHandler' works only with VisdomLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        for name, p in self.model.named_parameters():
            name = name.replace(".", "/")
            k = f"{tag_prefix}weights_{self.reduction.__name__}/{name}"
            v = self.reduction(p.data)
            self.add_scalar(logger, k, v, event_name, global_step)

        logger._save()


class GradsScalarHandler(BaseWeightsScalarHandler, _BaseVisDrawer):
    """Helper handler to log model's gradients as scalars.
    Handler iterates over the gradients of named parameters of the model, applies reduction function to each parameter
    produce a scalar and then logs the scalar.

    Args:
        model: model to log weights
        reduction: function to reduce parameters into scalar
        tag: common title for all produced plots. For example, "generator"
        show_legend: flag to show legend in the window

    Examples:
        .. code-block:: python

            from ignite.contrib.handlers.visdom_logger import *

            # Create a logger
            vd_logger = VisdomLogger()

            # Attach the logger to the trainer to log model's weights norm after each iteration
            vd_logger.attach(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                log_handler=GradsScalarHandler(model, reduction=torch.norm)
            )
    """

    def __init__(
        self, model: nn.Module, reduction: Callable = torch.norm, tag: Optional[str] = None, show_legend: bool = False
    ):
        super(GradsScalarHandler, self).__init__(model, reduction, tag)
        _BaseVisDrawer.__init__(self, show_legend=show_legend)

    def __call__(self, engine: Engine, logger: VisdomLogger, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, VisdomLogger):
            raise RuntimeError("Handler 'GradsScalarHandler' works only with VisdomLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue

            name = name.replace(".", "/")
            k = f"{tag_prefix}grads_{self.reduction.__name__}/{name}"
            v = self.reduction(p.grad)
            self.add_scalar(logger, k, v, event_name, global_step)

        logger._save()


class _DummyExecutor:
    class _DummyFuture:
        def __init__(self, result: Any) -> None:
            self._output = result

        def result(self) -> Any:
            return self._output

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def submit(self, fn: Callable, **kwargs: Any) -> "_DummyFuture":
        return _DummyExecutor._DummyFuture(fn(**kwargs))

    def shutdown(self, *args: Any, **kwargs: Any) -> None:
        pass
