import numbers
import warnings
from functools import partial
from typing import Any, Callable, cast, Dict, Iterable, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.distributed import DistributedSampler

# https://github.com/pytorch/ignite/issues/2773
try:
    from torch.optim.lr_scheduler import LRScheduler as PyTorchLRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as PyTorchLRScheduler

import ignite.distributed as idist
from ignite.contrib.handlers import (
    ClearMLLogger,
    global_step_from_engine,
    MLflowLogger,
    NeptuneLogger,
    PolyaxonLogger,
    ProgressBar,
    TensorboardLogger,
    VisdomLogger,
    WandBLogger,
)
from ignite.contrib.handlers.base_logger import BaseLogger
from ignite.contrib.metrics import GpuInfo
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, TerminateOnNan
from ignite.handlers.checkpoint import BaseSaveHandler
from ignite.handlers.param_scheduler import ParamScheduler
from ignite.metrics import RunningAverage
from ignite.metrics.metric import RunningBatchWise
from ignite.utils import deprecated


def setup_common_training_handlers(
    trainer: Engine,
    train_sampler: Optional[DistributedSampler] = None,
    to_save: Optional[Mapping] = None,
    save_every_iters: int = 1000,
    output_path: Optional[str] = None,
    lr_scheduler: Optional[Union[ParamScheduler, PyTorchLRScheduler]] = None,
    with_gpu_stats: bool = False,
    output_names: Optional[Iterable[str]] = None,
    with_pbars: bool = True,
    with_pbar_on_iters: bool = True,
    log_every_iters: int = 100,
    stop_on_nan: bool = True,
    clear_cuda_cache: bool = True,
    save_handler: Optional[Union[Callable, BaseSaveHandler]] = None,
    **kwargs: Any,
) -> None:
    """Helper method to setup trainer with common handlers (it also supports distributed configuration):

        - :class:`~ignite.handlers.terminate_on_nan.TerminateOnNan`
        - handler to setup learning rate scheduling
        - :class:`~ignite.handlers.checkpoint.ModelCheckpoint`
        - :class:`~ignite.metrics.RunningAverage` on `update_function` output
        - Two progress bars on epochs and optionally on iterations

    Args:
        trainer: trainer engine. Output of trainer's `update_function` should be a dictionary
            or sequence or a single tensor.
        train_sampler: Optional distributed sampler used to call
            `set_epoch` method on epoch started event.
        to_save: dictionary with objects to save in the checkpoint. This argument is passed to
            :class:`~ignite.handlers.checkpoint.Checkpoint` instance.
        save_every_iters: saving interval. By default, `to_save` objects are stored
            each 1000 iterations.
        output_path: output path to indicate where `to_save` objects are stored. Argument is mutually
            exclusive with ``save_handler``.
        lr_scheduler: learning rate scheduler
            as native torch LRScheduler or ignite's parameter scheduler.
        with_gpu_stats: if True, :class:`~ignite.contrib.metrics.GpuInfo` is attached to the
            trainer. This requires `pynvml` package to be installed.
        output_names: list of names associated with `update_function` output dictionary.
        with_pbars: if True, two progress bars on epochs and optionally on iterations are attached.
            Default, True.
        with_pbar_on_iters: if True, a progress bar on iterations is attached to the trainer.
            Default, True.
        log_every_iters: logging interval for :class:`~ignite.contrib.metrics.GpuInfo` and for
            epoch-wise progress bar. Default, 100.
        stop_on_nan: if True, :class:`~ignite.handlers.terminate_on_nan.TerminateOnNan` handler is added to the trainer.
            Default, True.
        clear_cuda_cache: if True, `torch.cuda.empty_cache()` is called every end of epoch.
            Default, True.
        save_handler: Method or callable
            class to use to store ``to_save``. See :class:`~ignite.handlers.checkpoint.Checkpoint` for more details.
            Argument is mutually exclusive with ``output_path``.
        kwargs: optional keyword args to be passed to construct :class:`~ignite.handlers.checkpoint.Checkpoint`.
    """

    if idist.get_world_size() > 1:
        _setup_common_distrib_training_handlers(
            trainer,
            train_sampler=train_sampler,
            to_save=to_save,
            save_every_iters=save_every_iters,
            output_path=output_path,
            lr_scheduler=lr_scheduler,
            with_gpu_stats=with_gpu_stats,
            output_names=output_names,
            with_pbars=with_pbars,
            with_pbar_on_iters=with_pbar_on_iters,
            log_every_iters=log_every_iters,
            stop_on_nan=stop_on_nan,
            clear_cuda_cache=clear_cuda_cache,
            save_handler=save_handler,
            **kwargs,
        )
    else:
        if train_sampler is not None and isinstance(train_sampler, DistributedSampler):
            warnings.warn(
                "Argument train_sampler is a distributed sampler,"
                " but either there is no distributed setting or world size is < 2. "
                "Train sampler argument will be ignored",
                UserWarning,
            )
        _setup_common_training_handlers(
            trainer,
            to_save=to_save,
            save_every_iters=save_every_iters,
            output_path=output_path,
            lr_scheduler=lr_scheduler,
            with_gpu_stats=with_gpu_stats,
            output_names=output_names,
            with_pbars=with_pbars,
            with_pbar_on_iters=with_pbar_on_iters,
            log_every_iters=log_every_iters,
            stop_on_nan=stop_on_nan,
            clear_cuda_cache=clear_cuda_cache,
            save_handler=save_handler,
            **kwargs,
        )


setup_common_distrib_training_handlers = setup_common_training_handlers


def _setup_common_training_handlers(
    trainer: Engine,
    to_save: Optional[Mapping] = None,
    save_every_iters: int = 1000,
    output_path: Optional[str] = None,
    lr_scheduler: Optional[Union[ParamScheduler, PyTorchLRScheduler]] = None,
    with_gpu_stats: bool = False,
    output_names: Optional[Iterable[str]] = None,
    with_pbars: bool = True,
    with_pbar_on_iters: bool = True,
    log_every_iters: int = 100,
    stop_on_nan: bool = True,
    clear_cuda_cache: bool = True,
    save_handler: Optional[Union[Callable, BaseSaveHandler]] = None,
    **kwargs: Any,
) -> None:
    if output_path is not None and save_handler is not None:
        raise ValueError(
            "Arguments output_path and save_handler are mutually exclusive. Please, define only one of them"
        )

    if stop_on_nan:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    if lr_scheduler is not None:
        if isinstance(lr_scheduler, PyTorchLRScheduler):
            trainer.add_event_handler(
                Events.ITERATION_COMPLETED, lambda engine: cast(PyTorchLRScheduler, lr_scheduler).step()
            )
        else:
            trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

    if torch.cuda.is_available() and clear_cuda_cache:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)

    if to_save is not None:
        if output_path is None and save_handler is None:
            raise ValueError(
                "If to_save argument is provided then output_path or save_handler arguments should be also defined"
            )
        if output_path is not None:
            save_handler = DiskSaver(dirname=output_path, require_empty=False)

        checkpoint_handler = Checkpoint(
            to_save, cast(Union[Callable, BaseSaveHandler], save_handler), filename_prefix="training", **kwargs
        )
        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=save_every_iters), checkpoint_handler)

    if with_gpu_stats:
        GpuInfo().attach(
            trainer, name="gpu", event_name=Events.ITERATION_COMPLETED(every=log_every_iters)  # type: ignore[arg-type]
        )

    if output_names is not None:

        def output_transform(x: Any, index: int, name: str) -> Any:
            if isinstance(x, Mapping):
                return x[name]
            elif isinstance(x, Sequence):
                return x[index]
            elif isinstance(x, (torch.Tensor, numbers.Number)):
                return x
            else:
                raise TypeError(
                    "Unhandled type of update_function's output. "
                    f"It should either mapping or sequence, but given {type(x)}"
                )

        for i, n in enumerate(output_names):
            RunningAverage(output_transform=partial(output_transform, index=i, name=n)).attach(
                trainer, n, usage=RunningBatchWise()
            )

    if with_pbars:
        if with_pbar_on_iters:
            ProgressBar(persist=False).attach(
                trainer, metric_names="all", event_name=Events.ITERATION_COMPLETED(every=log_every_iters)
            )

        ProgressBar(persist=True, bar_format="").attach(
            trainer, event_name=Events.EPOCH_STARTED, closing_event_name=Events.COMPLETED
        )


def _setup_common_distrib_training_handlers(
    trainer: Engine,
    train_sampler: Optional[DistributedSampler] = None,
    to_save: Optional[Mapping] = None,
    save_every_iters: int = 1000,
    output_path: Optional[str] = None,
    lr_scheduler: Optional[Union[ParamScheduler, PyTorchLRScheduler]] = None,
    with_gpu_stats: bool = False,
    output_names: Optional[Iterable[str]] = None,
    with_pbars: bool = True,
    with_pbar_on_iters: bool = True,
    log_every_iters: int = 100,
    stop_on_nan: bool = True,
    clear_cuda_cache: bool = True,
    save_handler: Optional[Union[Callable, BaseSaveHandler]] = None,
    **kwargs: Any,
) -> None:
    _setup_common_training_handlers(
        trainer,
        to_save=to_save,
        output_path=output_path,
        save_every_iters=save_every_iters,
        lr_scheduler=lr_scheduler,
        with_gpu_stats=with_gpu_stats,
        output_names=output_names,
        with_pbars=(idist.get_rank() == 0) and with_pbars,
        with_pbar_on_iters=with_pbar_on_iters,
        log_every_iters=log_every_iters,
        stop_on_nan=stop_on_nan,
        clear_cuda_cache=clear_cuda_cache,
        save_handler=save_handler,
        **kwargs,
    )

    if train_sampler is not None:
        if not isinstance(train_sampler, DistributedSampler):
            raise TypeError("Train sampler should be torch DistributedSampler and have `set_epoch` method")

        @trainer.on(Events.EPOCH_STARTED)
        def distrib_set_epoch(engine: Engine) -> None:
            train_sampler.set_epoch(engine.state.epoch - 1)


def empty_cuda_cache(_: Engine) -> None:
    torch.cuda.empty_cache()
    import gc

    gc.collect()


@deprecated(
    "0.4.0",
    "0.6.0",
    ("Please use instead: setup_tb_logging, setup_visdom_logging or setup_mlflow_logging etc.",),
    raise_exception=True,
)
def setup_any_logging(
    logger: BaseLogger,
    logger_module: Any,
    trainer: Engine,
    optimizers: Optional[Union[Optimizer, Dict[str, Optimizer], Dict[None, Optimizer]]],
    evaluators: Optional[Union[Engine, Dict[str, Engine]]],
    log_every_iters: int,
) -> None:
    pass


def _setup_logging(
    logger: BaseLogger,
    trainer: Engine,
    optimizers: Optional[Union[Optimizer, Dict[str, Optimizer], Dict[None, Optimizer]]],
    evaluators: Optional[Union[Engine, Dict[str, Engine]]],
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
    optimizers: Optional[Union[Optimizer, Dict[str, Optimizer]]] = None,
    evaluators: Optional[Union[Engine, Dict[str, Engine]]] = None,
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
        :class:`~ignite.contrib.handlers.tensorboard_logger.TensorboardLogger`
    """
    logger = TensorboardLogger(log_dir=output_path, **kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_visdom_logging(
    trainer: Engine,
    optimizers: Optional[Union[Optimizer, Dict[str, Optimizer]]] = None,
    evaluators: Optional[Union[Engine, Dict[str, Engine]]] = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> VisdomLogger:
    """Method to setup Visdom logging on trainer and a list of evaluators. Logged metrics are:

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
        :class:`~ignite.contrib.handlers.visdom_logger.VisdomLogger`
    """
    logger = VisdomLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_mlflow_logging(
    trainer: Engine,
    optimizers: Optional[Union[Optimizer, Dict[str, Optimizer]]] = None,
    evaluators: Optional[Union[Engine, Dict[str, Engine]]] = None,
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
        :class:`~ignite.contrib.handlers.mlflow_logger.MLflowLogger`
    """
    logger = MLflowLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_neptune_logging(
    trainer: Engine,
    optimizers: Optional[Union[Optimizer, Dict[str, Optimizer]]] = None,
    evaluators: Optional[Union[Engine, Dict[str, Engine]]] = None,
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
        :class:`~ignite.contrib.handlers.neptune_logger.NeptuneLogger`
    """
    logger = NeptuneLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_wandb_logging(
    trainer: Engine,
    optimizers: Optional[Union[Optimizer, Dict[str, Optimizer]]] = None,
    evaluators: Optional[Union[Engine, Dict[str, Engine]]] = None,
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
        :class:`~ignite.contrib.handlers.wandb_logger.WandBLogger`
    """
    logger = WandBLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_plx_logging(
    trainer: Engine,
    optimizers: Optional[Union[Optimizer, Dict[str, Optimizer]]] = None,
    evaluators: Optional[Union[Engine, Dict[str, Engine]]] = None,
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
        :class:`~ignite.contrib.handlers.polyaxon_logger.PolyaxonLogger`
    """
    logger = PolyaxonLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_clearml_logging(
    trainer: Engine,
    optimizers: Optional[Union[Optimizer, Dict[str, Optimizer]]] = None,
    evaluators: Optional[Union[Engine, Dict[str, Engine]]] = None,
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
        :class:`~ignite.contrib.handlers.clearml_logger.ClearMLLogger`
    """
    logger = ClearMLLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_trains_logging(
    trainer: Engine,
    optimizers: Optional[Union[Optimizer, Dict[str, Optimizer]]] = None,
    evaluators: Optional[Union[Engine, Dict[str, Engine]]] = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> ClearMLLogger:
    """``setup_trains_logging`` was renamed to :func:`~ignite.contrib.engines.common.setup_clearml_logging`."""
    warnings.warn("setup_trains_logging was renamed to setup_clearml_logging.")
    return setup_clearml_logging(trainer, optimizers, evaluators, log_every_iters, **kwargs)


get_default_score_fn = Checkpoint.get_default_score_fn


def gen_save_best_models_by_val_score(
    save_handler: Union[Callable, BaseSaveHandler],
    evaluator: Engine,
    models: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
    metric_name: str,
    n_saved: int = 3,
    trainer: Optional[Engine] = None,
    tag: str = "val",
    score_sign: float = 1.0,
    **kwargs: Any,
) -> Checkpoint:
    """Method adds a handler to ``evaluator`` to save ``n_saved`` of best models based on the metric
    (named by ``metric_name``) provided by ``evaluator`` (i.e. ``evaluator.state.metrics[metric_name]``).
    Models with highest metric value will be retained. The logic of how to store objects is delegated to
    ``save_handler``.

    Args:
        save_handler: Method or callable class to
            use to save engine and other provided objects. Function receives two objects: checkpoint as a dictionary
            and filename. If ``save_handler`` is callable class, it can
            inherit of :class:`~ignite.handlers.checkpoint.BaseSaveHandler` and optionally implement ``remove`` method
            to keep a fixed number of saved checkpoints. In case if user needs to save engine's checkpoint on a disk,
            ``save_handler`` can be defined with :class:`~ignite.handlers.DiskSaver`.
        evaluator: evaluation engine used to provide the score
        models: model or dictionary with the object to save. Objects should have
            implemented ``state_dict`` and ``load_state_dict`` methods.
        metric_name: metric name to use for score evaluation. This metric should be present in
            `evaluator.state.metrics`.
        n_saved: number of best models to store
        trainer: trainer engine to fetch the epoch when saving the best model.
        tag: score name prefix: `{tag}_{metric_name}`. By default, tag is "val".
        score_sign: sign of the score: 1.0 or -1.0. For error-like metrics, e.g. smaller is better,
            a negative score sign should be used (objects with larger score are retained). Default, 1.0.
        kwargs: optional keyword args to be passed to construct :class:`~ignite.handlers.checkpoint.Checkpoint`.

    Returns:
        A :class:`~ignite.handlers.checkpoint.Checkpoint` handler.
    """
    global_step_transform = None
    if trainer is not None:
        global_step_transform = global_step_from_engine(trainer)

    if isinstance(models, nn.Module):
        to_save: Dict[str, nn.Module] = {"model": models}
    else:
        to_save = models

    best_model_handler = Checkpoint(
        to_save,
        save_handler,
        filename_prefix="best",
        n_saved=n_saved,
        global_step_transform=global_step_transform,
        score_name=f"{tag}_{metric_name.lower()}",
        score_function=get_default_score_fn(metric_name, score_sign=score_sign),
        **kwargs,
    )
    evaluator.add_event_handler(Events.COMPLETED, best_model_handler)

    return best_model_handler


def save_best_model_by_val_score(
    output_path: str,
    evaluator: Engine,
    model: torch.nn.Module,
    metric_name: str,
    n_saved: int = 3,
    trainer: Optional[Engine] = None,
    tag: str = "val",
    score_sign: float = 1.0,
    **kwargs: Any,
) -> Checkpoint:
    """Method adds a handler to ``evaluator`` to save on a disk ``n_saved`` of best models based on the metric
    (named by ``metric_name``) provided by ``evaluator`` (i.e. ``evaluator.state.metrics[metric_name]``).
    Models with highest metric value will be retained.

    Args:
        output_path: output path to indicate where to save best models
        evaluator: evaluation engine used to provide the score
        model: model to store
        metric_name: metric name to use for score evaluation. This metric should be present in
            `evaluator.state.metrics`.
        n_saved: number of best models to store
        trainer: trainer engine to fetch the epoch when saving the best model.
        tag: score name prefix: `{tag}_{metric_name}`. By default, tag is "val".
        score_sign: sign of the score: 1.0 or -1.0. For error-like metrics, e.g. smaller is better,
            a negative score sign should be used (objects with larger score are retained). Default, 1.0.

        kwargs: optional keyword args to be passed to construct :class:`~ignite.handlers.checkpoint.Checkpoint`.

    Returns:
        A :class:`~ignite.handlers.checkpoint.Checkpoint` handler.
    """
    return gen_save_best_models_by_val_score(
        save_handler=DiskSaver(dirname=output_path, require_empty=False),
        evaluator=evaluator,
        models=model,
        metric_name=metric_name,
        n_saved=n_saved,
        trainer=trainer,
        tag=tag,
        score_sign=score_sign,
        **kwargs,
    )


def add_early_stopping_by_val_score(
    patience: int,
    evaluator: Engine,
    trainer: Engine,
    metric_name: str,
    score_sign: float = 1.0,
) -> EarlyStopping:
    """Method setups early stopping handler based on the score (named by `metric_name`) provided by `evaluator`.
    Metric value should increase in order to keep training and not early stop.

    Args:
        patience: number of events to wait if no improvement and then stop the training.
        evaluator: evaluation engine used to provide the score
        trainer: trainer engine to stop the run if no improvement.
        metric_name: metric name to use for score evaluation. This metric should be present in
            `evaluator.state.metrics`.
        score_sign: sign of the score: 1.0 or -1.0. For error-like metrics, e.g. smaller is better,
            a negative score sign should be used (objects with larger score are retained). Default, 1.0.

    Returns:
        A :class:`~ignite.handlers.early_stopping.EarlyStopping` handler.
    """
    es_handler = EarlyStopping(
        patience=patience, score_function=get_default_score_fn(metric_name, score_sign=score_sign), trainer=trainer
    )
    evaluator.add_event_handler(Events.COMPLETED, es_handler)

    return es_handler
