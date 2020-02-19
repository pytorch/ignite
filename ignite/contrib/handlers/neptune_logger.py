import os

import numbers
import tempfile
import warnings

import torch

import ignite
from ignite.engine import Events
from ignite.contrib.handlers.base_logger import (
    BaseLogger,
    BaseOptimizerParamsHandler,
    BaseOutputHandler,
    BaseWeightsScalarHandler,
    global_step_from_engine,
)

__all__ = [
    "NeptuneLogger",
    "OptimizerParamsHandler",
    "OutputHandler",
    "WeightsScalarHandler",
    "GradsScalarHandler",
    "global_step_from_engine",
]


class OutputHandler(BaseOutputHandler):
    """Helper handler to log engine's output and/or metrics

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.neptune_logger import *

            # Create a logger
            npt_logger = NeptuneLogger(api_token=os.environ["NEPTUNE_API_TOKEN"],
                                       project_name="USER_NAME/PROJECT_NAME",
                                       experiment_name="cnn-mnist", # Optional,
                                       params={"max_epochs": 10}, # Optional,
                                       tags=["pytorch-ignite","minst"] # Optional
                                       )

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer`:
            npt_logger.attach(evaluator,
                             log_handler=OutputHandler(tag="validation",
                                                       metric_names=["nll", "accuracy"],
                                                       global_step_transform=global_step_from_engine(trainer)),
                             event_name=Events.EPOCH_COMPLETED)

        Another example, where model is evaluated every 500 iterations:

        .. code-block:: python

            from ignite.contrib.handlers.neptune_logger import *

            @trainer.on(Events.ITERATION_COMPLETED(every=500))
            def evaluate(engine):
                evaluator.run(validation_set, max_epochs=1)

            npt_logger = NeptuneLogger(api_token=os.environ["NEPTUNE_API_TOKEN"],
                                       project_name="USER_NAME/PROJECT_NAME",
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

            npt_logger.attach(evaluator,
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
            :meth:`~ignite.contrib.handlers.neptune_logger.global_step_from_engine`.

    Note:

        Example of `global_step_transform`:

        .. code-block:: python

            def global_step_transform(engine, event_name):
                return engine.state.get_event_attrib_value(event_name)

    """

    def __init__(self, tag, metric_names=None, output_transform=None, another_engine=None, global_step_transform=None):
        super(OutputHandler, self).__init__(tag, metric_names, output_transform, another_engine, global_step_transform)

    def __call__(self, engine, logger, event_name):

        if not isinstance(logger, NeptuneLogger):
            raise RuntimeError("Handler 'OutputHandler' works only with NeptuneLogger")

        metrics = self._setup_output_metrics(engine)

        global_step = self.global_step_transform(engine, event_name)

        if not isinstance(global_step, int):
            raise TypeError(
                "global_step must be int, got {}."
                " Please check the output of global_step_transform.".format(type(global_step))
            )

        for key, value in metrics.items():
            if isinstance(value, numbers.Number) or isinstance(value, torch.Tensor) and value.ndimension() == 0:
                logger.experiment.log_metric("{}/{}".format(self.tag, key), x=global_step, y=value)
            elif isinstance(value, torch.Tensor) and value.ndimension() == 1:
                for i, v in enumerate(value):
                    logger.experiment.log_metric("{}/{}/{}".format(self.tag, key, i), x=global_step, y=v.item())
            else:
                warnings.warn("NeptuneLogger output_handler can not log " "metrics value type {}".format(type(value)))


class OptimizerParamsHandler(BaseOptimizerParamsHandler):
    """Helper handler to log optimizer parameters

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.neptune_logger import *

            # Create a logger
            npt_logger = NeptuneLogger(api_token=os.environ["NEPTUNE_API_TOKEN"],
                                       project_name="USER_NAME/PROJECT_NAME",
                                       experiment_name="cnn-mnist", # Optional,
                                       params={"max_epochs": 10}, # Optional,
                                       tags=["pytorch-ignite","minst"] # Optional
                                       )

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            npt_logger.attach(trainer,
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
        if not isinstance(logger, NeptuneLogger):
            raise RuntimeError("Handler 'OptimizerParamsHandler' works only with NeptuneLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        params = {
            "{}{}/group_{}".format(tag_prefix, self.param_name, i): float(param_group[self.param_name])
            for i, param_group in enumerate(self.optimizer.param_groups)
        }

        for k, v in params.items():
            logger.experiment.log_metric(k, x=global_step, y=v)


class WeightsScalarHandler(BaseWeightsScalarHandler):
    """Helper handler to log model's weights as scalars.
    Handler iterates over named parameters of the model, applies reduction function to each parameter
    produce a scalar and then logs the scalar.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.neptune_logger import *

            # Create a logger
            npt_logger = NeptuneLogger(api_token=os.environ["NEPTUNE_API_TOKEN"],
                                       project_name="USER_NAME/PROJECT_NAME",
                                       experiment_name="cnn-mnist", # Optional,
                                       params={"max_epochs": 10}, # Optional,
                                       tags=["pytorch-ignite","minst"] # Optional
                                       )

            # Attach the logger to the trainer to log model's weights norm after each iteration
            npt_logger.attach(trainer,
                             log_handler=WeightsScalarHandler(model, reduction=torch.norm),
                             event_name=Events.ITERATION_COMPLETED)

    Args:
        model (torch.nn.Module): model to log weights
        reduction (callable): function to reduce parameters into scalar
        tag (str, optional): common title for all produced plots. For example, 'generator'

    """

    def __init__(self, model, reduction=torch.norm, tag=None):
        super(WeightsScalarHandler, self).__init__(model, reduction, tag=tag)

    def __call__(self, engine, logger, event_name):

        if not isinstance(logger, NeptuneLogger):
            raise RuntimeError("Handler 'WeightsScalarHandler' works only with NeptuneLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue

            name = name.replace(".", "/")
            logger.experiment.log_metric(
                "{}weights_{}/{}".format(tag_prefix, self.reduction.__name__, name),
                x=global_step,
                y=self.reduction(p.data),
            )


class GradsScalarHandler(BaseWeightsScalarHandler):
    """Helper handler to log model's gradients as scalars.
    Handler iterates over the gradients of named parameters of the model, applies reduction function to each parameter
    produce a scalar and then logs the scalar.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.neptune_logger import *

            # Create a logger
            npt_logger = NeptuneLogger(api_token=os.environ["NEPTUNE_API_TOKEN"],
                                       project_name="USER_NAME/PROJECT_NAME",
                                       experiment_name="cnn-mnist", # Optional,
                                       params={"max_epochs": 10}, # Optional,
                                       tags=["pytorch-ignite","minst"] # Optional
                                       )

            # Attach the logger to the trainer to log model's weights norm after each iteration
            npt_logger.attach(trainer,
                             log_handler=GradsScalarHandler(model, reduction=torch.norm),
                             event_name=Events.ITERATION_COMPLETED)

    Args:
        model (torch.nn.Module): model to log weights
        reduction (callable): function to reduce parameters into scalar
        tag (str, optional): common title for all produced plots. For example, 'generator'

    """

    def __init__(self, model, reduction=torch.norm, tag=None):
        super(GradsScalarHandler, self).__init__(model, reduction, tag=tag)

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, NeptuneLogger):
            raise RuntimeError("Handler 'GradsScalarHandler' works only with NeptuneLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = "{}/".format(self.tag) if self.tag else ""
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue

            name = name.replace(".", "/")
            logger.experiment.log_metric(
                "{}grads_{}/{}".format(tag_prefix, self.reduction.__name__, name),
                x=global_step,
                y=self.reduction(p.grad),
            )


class NeptuneLogger(BaseLogger):
    """
    Neptune handler to log metrics, model/optimizer parameters, gradients during the training and validation.
    It can also log model checkpoints to Neptune server.

    .. code-block:: bash

        pip install neptune-client

    Args:
        api_token (str | None): Required in online mode. Neputne API token, found on https://neptune.ai.
            Read how to get your API key
            https://docs.neptune.ai/python-api/tutorials/get-started.html#copy-api-token.
        project_name (str): Required in online mode. Qualified name of a project in a form of
           "namespace/project_name" for example "tom/minst-classification".
           If None, the value of NEPTUNE_PROJECT environment variable will be taken.
           You need to create the project in https://neptune.ai first.
        offline_mode (bool): Optional default False. If offline_mode=True no logs will be send to neptune.
           Usually used for debug purposes.
        experiment_name (str, optional): Optional. Editable name of the experiment.
           Name is displayed in the experiment’s Details (Metadata section) and in experiments view as a column.
        upload_source_files (list, optional): Optional. List of source files to be uploaded.
           Must be list of str or single str. Uploaded sources are displayed in the experiment’s Source code tab.
           If None is passed, Python file from which experiment was created will be uploaded.
           Pass empty list (`[]`) to upload no files. Unix style pathname pattern expansion is supported.
           For example, you can pass `*.py` to upload all python source files from the current directory.
           For recursion lookup use `**/*.py` (for Python 3.5 and later). For more information see glob library.
        params (dict, optional): Optional. Parameters of the experiment. After experiment creation params are read-only.
           Parameters are displayed in the experiment’s Parameters section and each key-value pair can be
           viewed in experiments view as a column.
        properties (dict, optional): Optional default is `{}`. Properties of the experiment.
           They are editable after experiment is created. Properties are displayed in the experiment’s Details and
           each key-value pair can be viewed in experiments view as a column.
        tags (list, optional): Optional default `[]`. Must be list of str. Tags of the experiment.
           Tags are displayed in the experiment’s Details and can be viewed in experiments view as a column.

    Examples:

        .. code-block:: python

            from ignite.contrib.handlers.neptune_logger import *

            # Create a logger
            npt_logger = NeptuneLogger(api_token=os.environ["NEPTUNE_API_TOKEN"],
                                       project_name="USER_NAME/PROJECT_NAME",
                                       experiment_name="cnn-mnist", # Optional,
                                       params={"max_epochs": 10}, # Optional,
                                       tags=["pytorch-ignite","minst"] # Optional
                                       )

            # Attach the logger to the trainer to log training loss at each iteration
            npt_logger.attach(trainer,
                             log_handler=OutputHandler(tag="training", output_transform=lambda loss: {'loss': loss}),
                             event_name=Events.ITERATION_COMPLETED)

            # Attach the logger to the evaluator on the training dataset and log NLL, Accuracy metrics after each epoch
            # We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch
            # of the `trainer` instead of `train_evaluator`.
            npt_logger.attach(train_evaluator,
                             log_handler=OutputHandler(tag="training",
                                                       metric_names=["nll", "accuracy"],
                                                       global_step_transform=global_step_from_engine(trainer)),
                             event_name=Events.EPOCH_COMPLETED)

            # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
            # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
            # `trainer` instead of `evaluator`.
            npt_logger.attach(evaluator,
                             log_handler=OutputHandler(tag="validation",
                                                       metric_names=["nll", "accuracy"],
                                                       global_step_transform=global_step_from_engine(trainer)),
                             event_name=Events.EPOCH_COMPLETED)

            # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
            npt_logger.attach(trainer,
                             log_handler=OptimizerParamsHandler(optimizer),
                             event_name=Events.ITERATION_STARTED)

            # Attach the logger to the trainer to log model's weights norm after each iteration
            npt_logger.attach(trainer,
                             log_handler=WeightsScalarHandler(model),
                             event_name=Events.ITERATION_COMPLETED)

            # We need to close the logger when we are done
            npt_logger.close()

        Explore an experiment with neptune tracking here:
        https://ui.neptune.ai/o/neptune-ai/org/pytorch-ignite-integration/e/PYTOR-26/charts

        It is also possible to use the logger as context manager:

        .. code-block:: python

            from ignite.contrib.handlers.neptune_logger import *

            with npt_logger = NeptuneLogger(api_token=os.environ["NEPTUNE_API_TOKEN"],
                                            project_name="USER_NAME/PROJECT_NAME",
                                            experiment_name="cnn-mnist", # Optional,
                                            params={"max_epochs": 10}, # Optional,
                                            tags=["pytorch-ignite","minst"] # Optional
                                            ) as npt_logger:

                trainer = Engine(update_fn)
                # Attach the logger to the trainer to log training loss at each iteration
                npt_logger.attach(trainer,
                                 log_handler=OutputHandler(tag="training",
                                                           output_transform=lambda loss: {'loss': loss}),
                                 event_name=Events.ITERATION_COMPLETED)

    """

    def __init__(self, *args, **kwargs):
        try:
            import neptune
        except ImportError:
            raise RuntimeError(
                "This contrib module requires neptune-client to be installed. "
                "You may install neptune with command: \n pip install neptune-client \n"
            )

        if kwargs.get("offline_mode", False):
            self.mode = "offline"
            neptune.init(project_qualified_name="dry-run/project", backend=neptune.OfflineBackend())
        else:
            self.mode = "online"
            neptune.init(api_token=kwargs["api_token"], project_qualified_name=kwargs["project_name"])

        self._experiment_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["api_token", "project_name", "offline_mode"]
        }

        self.experiment = neptune.create_experiment(**self._experiment_kwargs)

    def close(self):
        self.experiment.stop()
