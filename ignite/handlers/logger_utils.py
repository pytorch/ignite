import warnings
from collections.abc import Mapping
from typing import Any

from torch.optim.optimizer import Optimizer

from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import (
    ClearMLLogger,
    MLflowLogger,
    NeptuneLogger,
    PolyaxonLogger,
    TensorboardLogger,
    VisdomLogger,
    WandBLogger,
    global_step_from_engine,
)
from ignite.handlers.base_logger import BaseLogger

__all__ = [
    "setup_tb_logging",
    "setup_visdom_logging",
    "setup_mlflow_logging",
    "setup_neptune_logging",
    "setup_wandb_logging",
    "setup_plx_logging",
    "setup_clearml_logging",
    "setup_trains_logging",
]


def _setup_logging(
    logger: BaseLogger,
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | dict[None, Optimizer] | None,
    evaluators: Engine | dict[str, Engine] | None,
    log_every_iters: int,
) -> None:
    if optimizers is not None:
        if not isinstance(optimizers, (Optimizer, Mapping)):
            raise TypeError("Argument optimizers should be either a single optimizer or a dictionary or optimizers")

    if evaluators is not None:
        if not isinstance(evaluators, (Engine, Mapping)):
            raise TypeError("Argument evaluators should be either a single engine or a dictionary or engines")

    if log_every_iters is None:
        log_every_iters = 1

    logger.attach_output_handler(
        trainer, event_name=Events.ITERATION_COMPLETED(every=log_every_iters), tag="training", metric_names="all"
    )

    if optimizers is not None:
        # Log optimizer parameters
        if isinstance(optimizers, Optimizer):
            optimizers = {None: optimizers}

        for k, optimizer in optimizers.items():
            logger.attach_opt_params_handler(
                trainer, Events.ITERATION_STARTED(every=log_every_iters), optimizer, param_name="lr", tag=k
            )

    if evaluators is not None:
        # Log evaluation metrics
        if isinstance(evaluators, Engine):
            evaluators = {"validation": evaluators}

        event_name = Events.ITERATION_COMPLETED if isinstance(logger, WandBLogger) else None
        gst = global_step_from_engine(trainer, custom_event_name=event_name)
        for k, evaluator in evaluators.items():
            logger.attach_output_handler(
                evaluator, event_name=Events.COMPLETED, tag=k, metric_names="all", global_step_transform=gst
            )


def setup_tb_logging(
    output_path: str,
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | None = None,
    evaluators: Engine | dict[str, Engine] | None = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> TensorboardLogger:
    """Method to setup TensorBoard logging on trainer and a list of evaluators. Logged metrics are:

        - Training metrics, e.g. running average loss values
        - Learning rate(s)
        - Evaluation metrics

    Args:
        output_path: logging directory path
        trainer: trainer engine
        optimizers: single or dictionary of
            torch optimizers. If a dictionary, keys are used as tags arguments for logging.
        evaluators: single or dictionary of evaluators. If a dictionary,
            keys are used as tags arguments for logging.
        log_every_iters: interval for loggers attached to iteration events. To log every iteration,
            value can be set to 1 or None.
        kwargs: optional keyword args to be passed to construct the logger.

    Returns:
        :class:`~ignite.handlers.tensorboard_logger.TensorboardLogger`

    Examples:
        .. code-block:: python

            from ignite.handlers.logger_utils import setup_tb_logging
            # Assume `trainer`, `evaluator`, and `optimizer` are already defined
            tb_logger = setup_tb_logging(
                output_path="experiments/tb_logs",
                trainer=trainer,
                optimizers=optimizer,
                evaluators={"validation": evaluator},
                log_every_iters=100
            )

            # Logger instance can be closed
            tb_logger.close()
    """
    logger = TensorboardLogger(log_dir=output_path, **kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_visdom_logging(
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | None = None,
    evaluators: Engine | dict[str, Engine] | None = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> VisdomLogger:
    """Method to setup Visdom logging on trainer and a list of evaluators. Logged metrics are:

        - Training metrics, e.g. running average loss values
        - Learning rate(s)
        - Evaluation metrics

    .. warning::

        This function uses VisdomLogger which is currently untested due to the visdom package being
        unmaintained and difficult to install with modern Python packages. Use at your own risk.

    Args:
        trainer: trainer engine
        optimizers: single or dictionary of
            torch optimizers. If a dictionary, keys are used as tags arguments for logging.
        evaluators: single or dictionary of evaluators. If a dictionary,
            keys are used as tags arguments for logging.
        log_every_iters: interval for loggers attached to iteration events. To log every iteration,
            value can be set to 1 or None.
        kwargs: optional keyword args to be passed to construct the logger.

    Returns:
        :class:`~ignite.handlers.visdom_logger.VisdomLogger`

    Examples:
        .. code-block:: python

            from ignite.handlers.logger_utils import setup_visdom_logging
            # Assume `trainer`, `evaluator`, and `optimizer` are already defined
            vd_logger = setup_visdom_logging(
                trainer=trainer,
                optimizers=optimizer,
                evaluators={"validation": evaluator},
                log_every_iters=100
            )

            # Logger instance can be closed
            vd_logger.close()
    """
    logger = VisdomLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_mlflow_logging(
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | None = None,
    evaluators: Engine | dict[str, Engine] | None = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> MLflowLogger:
    """Method to setup MLflow logging on trainer and a list of evaluators. Logged metrics are:

        - Training metrics, e.g. running average loss values
        - Learning rate(s)
        - Evaluation metrics

    Args:
        trainer: trainer engine
        optimizers: single or dictionary of
            torch optimizers. If a dictionary, keys are used as tags arguments for logging.
        evaluators: single or dictionary of evaluators. If a dictionary,
            keys are used as tags arguments for logging.
        log_every_iters: interval for loggers attached to iteration events. To log every iteration,
            value can be set to 1 or None.
        kwargs: optional keyword args to be passed to construct the logger.

    Returns:
        :class:`~ignite.handlers.mlflow_logger.MLflowLogger`

    Examples:
        .. code-block:: python

            from ignite.handlers.logger_utils import setup_mlflow_logging
            # Assume `trainer`, `evaluator`, and `optimizer` are already defined
            mlflow_logger = setup_mlflow_logging(
                trainer=trainer,
                optimizers=optimizer,
                evaluators={"validation": evaluator},
                log_every_iters=100
            )

            # Logger instance can be closed
            mlflow_logger.close()
    """
    logger = MLflowLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_neptune_logging(
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | None = None,
    evaluators: Engine | dict[str, Engine] | None = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> NeptuneLogger:
    """Method to setup Neptune logging on trainer and a list of evaluators. Logged metrics are:

        - Training metrics, e.g. running average loss values
        - Learning rate(s)
        - Evaluation metrics

    Args:
        trainer: trainer engine
        optimizers: single or dictionary of
            torch optimizers. If a dictionary, keys are used as tags arguments for logging.
        evaluators: single or dictionary of evaluators. If a dictionary,
            keys are used as tags arguments for logging.
        log_every_iters: interval for loggers attached to iteration events. To log every iteration,
            value can be set to 1 or None.
        kwargs: optional keyword args to be passed to construct the logger.

    Returns:
        :class:`~ignite.handlers.neptune_logger.NeptuneLogger`

    Examples:
        .. code-block:: python

            from ignite.handlers.logger_utils import setup_neptune_logging
            # Assume `trainer`, `evaluator`, and `optimizer` are already defined
            neptune_logger = setup_neptune_logging(
                trainer=trainer,
                optimizers=optimizer,
                evaluators={"validation": evaluator},
                log_every_iters=100
            )

            # Logger instance can be closed
            neptune_logger.close()
    """
    logger = NeptuneLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_wandb_logging(
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | None = None,
    evaluators: Engine | dict[str, Engine] | None = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> WandBLogger:
    """Method to setup WandB logging on trainer and a list of evaluators. Logged metrics are:

        - Training metrics, e.g. running average loss values
        - Learning rate(s)
        - Evaluation metrics

    Args:
        trainer: trainer engine
        optimizers: single or dictionary of
            torch optimizers. If a dictionary, keys are used as tags arguments for logging.
        evaluators: single or dictionary of evaluators. If a dictionary,
            keys are used as tags arguments for logging.
        log_every_iters: interval for loggers attached to iteration events. To log every iteration,
            value can be set to 1 or None.
        kwargs: optional keyword args to be passed to construct the logger.

    Returns:
        :class:`~ignite.handlers.wandb_logger.WandBLogger`

    Examples:
        .. code-block:: python

            from ignite.handlers.logger_utils import setup_wandb_logging
            # Assume `trainer`, `evaluator`, and `optimizer` are already defined
            wandb_logger = setup_wandb_logging(
                trainer=trainer,
                optimizers=optimizer,
                evaluators={"validation": evaluator},
                log_every_iters=100
            )

            # Logger instance can be closed
            wandb_logger.close()
    """
    logger = WandBLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_plx_logging(
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | None = None,
    evaluators: Engine | dict[str, Engine] | None = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> PolyaxonLogger:
    """Method to setup Polyaxon logging on trainer and a list of evaluators. Logged metrics are:

        - Training metrics, e.g. running average loss values
        - Learning rate(s)
        - Evaluation metrics

    Args:
        trainer: trainer engine
        optimizers: single or dictionary of
            torch optimizers. If a dictionary, keys are used as tags arguments for logging.
        evaluators: single or dictionary of evaluators. If a dictionary,
            keys are used as tags arguments for logging.
        log_every_iters: interval for loggers attached to iteration events. To log every iteration,
            value can be set to 1 or None.
        kwargs: optional keyword args to be passed to construct the logger.

    Returns:
        :class:`~ignite.handlers.polyaxon_logger.PolyaxonLogger`

    Examples:
        .. code-block:: python

            from ignite.handlers.logger_utils import setup_plx_logging
            # Assume `trainer`, `evaluator`, and `optimizer` are already defined
            plx_logger = setup_plx_logging(
                trainer=trainer,
                optimizers=optimizer,
                evaluators={"validation": evaluator},
                log_every_iters=100
            )

            # Logger instance can be closed
            plx_logger.close()
    """
    logger = PolyaxonLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_clearml_logging(
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | None = None,
    evaluators: Engine | dict[str, Engine] | None = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> ClearMLLogger:
    """Method to setup ClearML logging on trainer and a list of evaluators. Logged metrics are:

        - Training metrics, e.g. running average loss values
        - Learning rate(s)
        - Evaluation metrics

    Args:
        trainer: trainer engine
        optimizers: single or dictionary of
            torch optimizers. If a dictionary, keys are used as tags arguments for logging.
        evaluators: single or dictionary of evaluators. If a dictionary,
            keys are used as tags arguments for logging.
        log_every_iters: interval for loggers attached to iteration events. To log every iteration,
            value can be set to 1 or None.
        kwargs: optional keyword args to be passed to construct the logger.

    Returns:
        :class:`~ignite.handlers.clearml_logger.ClearMLLogger`

    Examples:
        .. code-block:: python

            from ignite.handlers.logger_utils import setup_clearml_logging
            # Assume `trainer`, `evaluator`, and `optimizer` are already defined
            clearml_logger = setup_clearml_logging(
                trainer=trainer,
                optimizers=optimizer,
                evaluators={"validation": evaluator},
                log_every_iters=100
            )

            # Logger instance can be closed
            clearml_logger.close()
    """
    logger = ClearMLLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_trains_logging(
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | None = None,
    evaluators: Engine | dict[str, Engine] | None = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> ClearMLLogger:
    """``setup_trains_logging`` was renamed to :func:`~ignite.handlers.logger_utils.setup_clearml_logging`."""
    warnings.warn("setup_trains_logging was renamed to setup_clearml_logging.")
    return setup_clearml_logging(trainer, optimizers, evaluators, log_every_iters, **kwargs)
