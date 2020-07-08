import numbers
import warnings
from collections.abc import Mapping, Sequence
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler

import ignite.distributed as idist
from ignite.contrib.handlers import (
    LRScheduler,
    MLflowLogger,
    NeptuneLogger,
    PolyaxonLogger,
    ProgressBar,
    TensorboardLogger,
    TrainsLogger,
    VisdomLogger,
    WandBLogger,
    global_step_from_engine,
)
from ignite.contrib.metrics import GpuInfo
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, TerminateOnNan
from ignite.metrics import RunningAverage


def setup_common_training_handlers(
    trainer,
    train_sampler=None,
    to_save=None,
    save_every_iters=1000,
    output_path=None,
    lr_scheduler=None,
    with_gpu_stats=False,
    output_names=None,
    with_pbars=True,
    with_pbar_on_iters=True,
    log_every_iters=100,
    device=None,
    stop_on_nan=True,
    clear_cuda_cache=True,
    save_handler=None,
    **kwargs
):
    """Helper method to setup trainer with common handlers (it also supports distributed configuration):
        - :class:`~ignite.handlers.TerminateOnNan`
        - handler to setup learning rate scheduling
        - :class:`~ignite.handlers.ModelCheckpoint`
        - :class:`~ignite.metrics.RunningAverage` on `update_function` output
        - Two progress bars on epochs and optionally on iterations

    Args:
        trainer (Engine): trainer engine. Output of trainer's `update_function` should be a dictionary
            or sequence or a single tensor.
        train_sampler (torch.utils.data.DistributedSampler, optional): Optional distributed sampler used to call
            `set_epoch` method on epoch started event.
        to_save (dict, optional): dictionary with objects to save in the checkpoint. This argument is passed to
            :class:`~ignite.handlers.Checkpoint` instance.
        save_every_iters (int, optional): saving interval. By default, `to_save` objects are stored
            each 1000 iterations.
        output_path (str, optional): output path to indicate where `to_save` objects are stored. Argument is mutually
            exclusive with ``save_handler``.
        lr_scheduler (ParamScheduler or subclass of `torch.optim.lr_scheduler._LRScheduler`): learning rate scheduler
            as native torch LRScheduler or ignite's parameter scheduler.
        with_gpu_stats (bool, optional): if True, :class:`~ignite.contrib.metrics.handlers.GpuInfo` is attached to the
            trainer. This requires `pynvml` package to be installed.
        output_names (list/tuple, optional): list of names associated with `update_function` output dictionary.
        with_pbars (bool, optional): if True, two progress bars on epochs and optionally on iterations are attached.
            Default, True.
        with_pbar_on_iters (bool, optional): if True, a progress bar on iterations is attached to the trainer.
            Default, True.
        log_every_iters (int, optional): logging interval for :class:`~ignite.contrib.metrics.handlers.GpuInfo` and for
            epoch-wise progress bar. Default, 100.
        stop_on_nan (bool, optional): if True, :class:`~ignite.handlers.TerminateOnNan` handler is added to the trainer.
            Default, True.
        clear_cuda_cache (bool, optional): if True, `torch.cuda.empty_cache()` is called every end of epoch.
            Default, True.
        save_handler (callable or :class:`~ignite.handlers.checkpoint.BaseSaveHandler`, optional): Method or callable
            class to use to store ``to_save``. See :class:`~ignite.handlers.checkpoint.Checkpoint` for more details.
            Argument is mutually exclusive with ``output_path``.
        **kwargs: optional keyword args to be passed to construct :class:`~ignite.handlers.checkpoint.Checkpoint`.
        device (str of torch.device, optional): deprecated argument, it will be removed in v0.5.0.
    """
    if device is not None:
        warnings.warn("Argument device is unused and deprecated. It will be removed in v0.5.0")

    _kwargs = dict(
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
    )
    _kwargs.update(kwargs)

    if idist.get_world_size() > 1:
        _setup_common_distrib_training_handlers(trainer, train_sampler=train_sampler, **_kwargs)
    else:
        if train_sampler is not None and isinstance(train_sampler, DistributedSampler):
            warnings.warn(
                "Argument train_sampler is a distributed sampler,"
                " but either there is no distributed setting or world size is < 2. "
                "Train sampler argument will be ignored",
                UserWarning,
            )
        _setup_common_training_handlers(trainer, **_kwargs)


setup_common_distrib_training_handlers = setup_common_training_handlers


def _setup_common_training_handlers(
    trainer,
    to_save=None,
    save_every_iters=1000,
    output_path=None,
    lr_scheduler=None,
    with_gpu_stats=False,
    output_names=None,
    with_pbars=True,
    with_pbar_on_iters=True,
    log_every_iters=100,
    stop_on_nan=True,
    clear_cuda_cache=True,
    save_handler=None,
    **kwargs
):
    if output_path is not None and save_handler is not None:
        raise ValueError(
            "Arguments output_path and save_handler are mutually exclusive. Please, define only one of them"
        )

    if stop_on_nan:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    if lr_scheduler is not None:
        if isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
            trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda engine: lr_scheduler.step())
        elif isinstance(lr_scheduler, LRScheduler):
            trainer.add_event_handler(Events.ITERATION_COMPLETED, lr_scheduler)
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

        checkpoint_handler = Checkpoint(to_save, save_handler, filename_prefix="training", **kwargs)
        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=save_every_iters), checkpoint_handler)

    if with_gpu_stats:
        GpuInfo().attach(trainer, name="gpu", event_name=Events.ITERATION_COMPLETED(every=log_every_iters))

    if output_names is not None:

        def output_transform(x, index, name):
            if isinstance(x, Mapping):
                return x[name]
            elif isinstance(x, Sequence):
                return x[index]
            elif isinstance(x, (torch.Tensor, numbers.Number)):
                return x
            else:
                raise TypeError(
                    "Unhandled type of update_function's output. "
                    "It should either mapping or sequence, but given {}".format(type(x))
                )

        for i, n in enumerate(output_names):
            RunningAverage(output_transform=partial(output_transform, index=i, name=n), epoch_bound=False).attach(
                trainer, n
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
    trainer,
    train_sampler=None,
    to_save=None,
    save_every_iters=1000,
    output_path=None,
    lr_scheduler=None,
    with_gpu_stats=False,
    output_names=None,
    with_pbars=True,
    with_pbar_on_iters=True,
    log_every_iters=100,
    stop_on_nan=True,
    clear_cuda_cache=True,
    save_handler=None,
    **kwargs
):

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
        def distrib_set_epoch(engine):
            train_sampler.set_epoch(engine.state.epoch - 1)


def empty_cuda_cache(_):
    torch.cuda.empty_cache()
    import gc

    gc.collect()


def setup_any_logging(logger, logger_module, trainer, optimizers, evaluators, log_every_iters):
    raise DeprecationWarning(
        "ignite.contrib.engines.common.setup_any_logging is deprecated since 0.4.0. "
        "Please use ignite.contrib.engines.common._setup_logging instead."
    )


def _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters):
    if optimizers is not None:
        from torch.optim.optimizer import Optimizer

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


def setup_tb_logging(output_path, trainer, optimizers=None, evaluators=None, log_every_iters=100, **kwargs):
    """Method to setup TensorBoard logging on trainer and a list of evaluators. Logged metrics are:
        - Training metrics, e.g. running average loss values
        - Learning rate(s)
        - Evaluation metrics

    Args:
        output_path (str): logging directory path
        trainer (Engine): trainer engine
        optimizers (torch.optim.Optimizer or dict of torch.optim.Optimizer, optional): single or dictionary of
            torch optimizers. If a dictionary, keys are used as tags arguments for logging.
        evaluators (Engine or dict of Engine, optional): single or dictionary of evaluators. If a dictionary,
            keys are used as tags arguments for logging.
        log_every_iters (int, optional): interval for loggers attached to iteration events. To log every iteration,
            value can be set to 1 or None.
        **kwargs: optional keyword args to be passed to construct the logger.

    Returns:
        TensorboardLogger
    """
    logger = TensorboardLogger(log_dir=output_path, **kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_visdom_logging(trainer, optimizers=None, evaluators=None, log_every_iters=100, **kwargs):
    """Method to setup Visdom logging on trainer and a list of evaluators. Logged metrics are:
        - Training metrics, e.g. running average loss values
        - Learning rate(s)
        - Evaluation metrics

    Args:
        trainer (Engine): trainer engine
        optimizers (torch.optim.Optimizer or dict of torch.optim.Optimizer, optional): single or dictionary of
            torch optimizers. If a dictionary, keys are used as tags arguments for logging.
        evaluators (Engine or dict of Engine, optional): single or dictionary of evaluators. If a dictionary,
            keys are used as tags arguments for logging.
        log_every_iters (int, optional): interval for loggers attached to iteration events. To log every iteration,
            value can be set to 1 or None.
        **kwargs: optional keyword args to be passed to construct the logger.

    Returns:
        VisdomLogger
    """
    logger = VisdomLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_mlflow_logging(trainer, optimizers=None, evaluators=None, log_every_iters=100, **kwargs):
    """Method to setup MLflow logging on trainer and a list of evaluators. Logged metrics are:
        - Training metrics, e.g. running average loss values
        - Learning rate(s)
        - Evaluation metrics

    Args:
        trainer (Engine): trainer engine
        optimizers (torch.optim.Optimizer or dict of torch.optim.Optimizer, optional): single or dictionary of
            torch optimizers. If a dictionary, keys are used as tags arguments for logging.
        evaluators (Engine or dict of Engine, optional): single or dictionary of evaluators. If a dictionary,
            keys are used as tags arguments for logging.
        log_every_iters (int, optional): interval for loggers attached to iteration events. To log every iteration,
            value can be set to 1 or None.
        **kwargs: optional keyword args to be passed to construct the logger.

    Returns:
        MLflowLogger
    """
    logger = MLflowLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_neptune_logging(trainer, optimizers=None, evaluators=None, log_every_iters=100, **kwargs):
    """Method to setup Neptune logging on trainer and a list of evaluators. Logged metrics are:
        - Training metrics, e.g. running average loss values
        - Learning rate(s)
        - Evaluation metrics

    Args:
        trainer (Engine): trainer engine
        optimizers (torch.optim.Optimizer or dict of torch.optim.Optimizer, optional): single or dictionary of
            torch optimizers. If a dictionary, keys are used as tags arguments for logging.
        evaluators (Engine or dict of Engine, optional): single or dictionary of evaluators. If a dictionary,
            keys are used as tags arguments for logging.
        log_every_iters (int, optional): interval for loggers attached to iteration events. To log every iteration,
            value can be set to 1 or None.
        **kwargs: optional keyword args to be passed to construct the logger.

    Returns:
        NeptuneLogger
    """
    logger = NeptuneLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_wandb_logging(trainer, optimizers=None, evaluators=None, log_every_iters=100, **kwargs):
    """Method to setup WandB logging on trainer and a list of evaluators. Logged metrics are:
        - Training metrics, e.g. running average loss values
        - Learning rate(s)
        - Evaluation metrics

    Args:
        trainer (Engine): trainer engine
        optimizers (torch.optim.Optimizer or dict of torch.optim.Optimizer, optional): single or dictionary of
            torch optimizers. If a dictionary, keys are used as tags arguments for logging.
        evaluators (Engine or dict of Engine, optional): single or dictionary of evaluators. If a dictionary,
            keys are used as tags arguments for logging.
        log_every_iters (int, optional): interval for loggers attached to iteration events. To log every iteration,
            value can be set to 1 or None.
        **kwargs: optional keyword args to be passed to construct the logger.

    Returns:
        WandBLogger
    """
    logger = WandBLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_plx_logging(trainer, optimizers=None, evaluators=None, log_every_iters=100, **kwargs):
    """Method to setup Polyaxon logging on trainer and a list of evaluators. Logged metrics are:
        - Training metrics, e.g. running average loss values
        - Learning rate(s)
        - Evaluation metrics

    Args:
        trainer (Engine): trainer engine
        optimizers (torch.optim.Optimizer or dict of torch.optim.Optimizer, optional): single or dictionary of
            torch optimizers. If a dictionary, keys are used as tags arguments for logging.
        evaluators (Engine or dict of Engine, optional): single or dictionary of evaluators. If a dictionary,
            keys are used as tags arguments for logging.
        log_every_iters (int, optional): interval for loggers attached to iteration events. To log every iteration,
            value can be set to 1 or None.
        **kwargs: optional keyword args to be passed to construct the logger.

    Returns:
        PolyaxonLogger
    """
    logger = PolyaxonLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_trains_logging(trainer, optimizers=None, evaluators=None, log_every_iters=100, **kwargs):
    """Method to setup Trains logging on trainer and a list of evaluators. Logged metrics are:
        - Training metrics, e.g. running average loss values
        - Learning rate(s)
        - Evaluation metrics

    Args:
        trainer (Engine): trainer engine
        optimizers (torch.optim.Optimizer or dict of torch.optim.Optimizer, optional): single or dictionary of
            torch optimizers. If a dictionary, keys are used as tags arguments for logging.
        evaluators (Engine or dict of Engine, optional): single or dictionary of evaluators. If a dictionary,
            keys are used as tags arguments for logging.
        log_every_iters (int, optional): interval for loggers attached to iteration events. To log every iteration,
            value can be set to 1 or None.
        **kwargs: optional keyword args to be passed to construct the logger.

    Returns:
        TrainsLogger
    """
    logger = TrainsLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def get_default_score_fn(metric_name):
    def wrapper(engine):
        score = engine.state.metrics[metric_name]
        return score

    return wrapper


def gen_save_best_models_by_val_score(
    save_handler, evaluator, models, metric_name, n_saved=3, trainer=None, tag="val", **kwargs
):
    """Method adds a handler to ``evaluator`` to save ``n_saved`` of best models based on the metric
    (named by ``metric_name``) provided by ``evaluator`` (i.e. ``evaluator.state.metrics[metric_name]``).
    The logic of how to store objects is delegated to ``save_handler``.

    Args:
        save_handler (callable or :class:`~ignite.handlers.checkpoint.BaseSaveHandler`): Method or callable class to
            use to save engine and other provided objects. Function receives two objects: checkpoint as a dictionary
            and filename. If ``save_handler`` is callable class, it can
            inherit of :class:`~ignite.handlers.checkpoint.BaseSaveHandler` and optionally implement ``remove`` method
            to keep a fixed number of saved checkpoints. In case if user needs to save engine's checkpoint on a disk,
            ``save_handler`` can be defined with :class:`~ignite.handlers.DiskSaver`.
        evaluator (Engine): evaluation engine used to provide the score
        models (nn.Module or Mapping): model or dictionary with the object to save. Objects should have
            implemented ``state_dict`` and ``load_state_dict`` methods.
        metric_name (str): metric name to use for score evaluation. This metric should be present in
            `evaluator.state.metrics`.
        n_saved (int, optional): number of best models to store
        trainer (Engine, optional): trainer engine to fetch the epoch when saving the best model.
        tag (str, optional): score name prefix: `{tag}_{metric_name}`. By default, tag is "val".
        **kwargs: optional keyword args to be passed to construct :class:`~ignite.handlers.checkpoint.Checkpoint`.

    Returns:
        A :class:`~ignite.handlers.checkpoint.Checkpoint` handler.
    """
    global_step_transform = None
    if trainer is not None:
        global_step_transform = global_step_from_engine(trainer)

    to_save = models
    if isinstance(models, nn.Module):
        to_save = {"model": models}

    best_model_handler = Checkpoint(
        to_save,
        save_handler,
        filename_prefix="best",
        n_saved=n_saved,
        global_step_transform=global_step_transform,
        score_name="{}_{}".format(tag, metric_name.lower()),
        score_function=get_default_score_fn(metric_name),
        **kwargs,
    )
    evaluator.add_event_handler(
        Events.COMPLETED, best_model_handler,
    )

    return best_model_handler


def save_best_model_by_val_score(
    output_path, evaluator, model, metric_name, n_saved=3, trainer=None, tag="val", **kwargs
):
    """Method adds a handler to ``evaluator`` to save on a disk ``n_saved`` of best models based on the metric
    (named by ``metric_name``) provided by ``evaluator`` (i.e. ``evaluator.state.metrics[metric_name]``).

    Args:
        output_path (str): output path to indicate where to save best models
        evaluator (Engine): evaluation engine used to provide the score
        model (nn.Module): model to store
        metric_name (str): metric name to use for score evaluation. This metric should be present in
            `evaluator.state.metrics`.
        n_saved (int, optional): number of best models to store
        trainer (Engine, optional): trainer engine to fetch the epoch when saving the best model.
        tag (str, optional): score name prefix: `{tag}_{metric_name}`. By default, tag is "val".
        **kwargs: optional keyword args to be passed to construct :class:`~ignite.handlers.checkpoint.Checkpoint`.

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
        **kwargs,
    )


def add_early_stopping_by_val_score(patience, evaluator, trainer, metric_name):
    """Method setups early stopping handler based on the score (named by `metric_name`) provided by `evaluator`.

    Args:
        patience (int): number of events to wait if no improvement and then stop the training.
        evaluator (Engine): evaluation engine used to provide the score
        trainer (Engine): trainer engine to stop the run if no improvement.
        metric_name (str): metric name to use for score evaluation. This metric should be present in
            `evaluator.state.metrics`.

    Returns:
        A :class:`~ignite.handlers.early_stopping.EarlyStopping` handler.
    """
    es_handler = EarlyStopping(patience=patience, score_function=get_default_score_fn(metric_name), trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, es_handler)

    return es_handler
