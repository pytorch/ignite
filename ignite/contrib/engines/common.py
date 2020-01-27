from functools import partial
import warnings

from collections.abc import Sequence, Mapping

import torch
import torch.distributed as dist

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import TerminateOnNan, ModelCheckpoint, EarlyStopping
from ignite.contrib.metrics import GpuInfo
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
import ignite.contrib.handlers.tensorboard_logger as tb_logger_module
from ignite.contrib.handlers import MLflowLogger
import ignite.contrib.handlers.mlflow_logger as mlflow_logger_module
from ignite.contrib.handlers import PolyaxonLogger
import ignite.contrib.handlers.polyaxon_logger as polyaxon_logger_module


def setup_common_training_handlers(trainer, train_sampler=None,
                                   to_save=None, save_every_iters=1000, output_path=None,
                                   lr_scheduler=None, with_gpu_stats=False,
                                   output_names=None, with_pbars=True, with_pbar_on_iters=True, log_every_iters=100,
                                   device='cuda'):
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
        to_save (dict, optional): dictionary with objects to save in the checkpoint. This is used with
            :class:`~ignite.handlers.ModelCheckpoint`.
        save_every_iters (int, optional): saving interval. By default, `to_save` objects are stored
            each 1000 iterations.
        output_path (str, optional): output path to indicate where `to_save` objects are stored.
        lr_scheduler (ParamScheduler or subclass of `torch.optim.lr_scheduler._LRScheduler`): learning rate scheduler
            as native torch LRScheduler or ignite's parameter scheduler.
        with_gpu_stats (bool, optional): if True, :class:`~ignite.contrib.metrics.handlers.GpuInfo` is attached to the
            trainer. This requires `pynvml` package to be installed.
        output_names (list/tuple): list of names associated with `update_function` output dictionary.
        with_pbars (bool, optional): if True, two progress bars on epochs and optionally on iterations are attached
        with_pbar_on_iters (bool, optional): if True, a progress bar on iterations is attached to the trainer.
        log_every_iters (int, optional): logging interval for :class:`~ignite.contrib.metrics.handlers.GpuInfo` and for
            epoch-wise progress bar.
        device (str of torch.device, optional): Optional device specification in case of distributed computation usage.
    """
    kwargs = dict(to_save=to_save,
                  save_every_iters=save_every_iters, output_path=output_path,
                  lr_scheduler=lr_scheduler, with_gpu_stats=with_gpu_stats,
                  output_names=output_names, with_pbars=with_pbars,
                  with_pbar_on_iters=with_pbar_on_iters,
                  log_every_iters=log_every_iters, device=device)
    if dist.is_available() and dist.is_initialized():
        return _setup_common_distrib_training_handlers(trainer, train_sampler=train_sampler, **kwargs)
    else:
        if train_sampler is not None:
            warnings.warn("Argument train_sampler distributed sampler used to call `set_epoch` method on epoch "
                          "started event, but no distributed setting detected", UserWarning)
        return _setup_common_training_handlers(trainer, **kwargs)


setup_common_distrib_training_handlers = setup_common_training_handlers


def _setup_common_training_handlers(trainer,
                                    to_save=None, save_every_iters=1000, output_path=None,
                                    lr_scheduler=None, with_gpu_stats=True,
                                    output_names=None, with_pbars=True, with_pbar_on_iters=True,
                                    log_every_iters=100, device='cuda'):
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    if lr_scheduler is not None:
        if isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
            trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda engine: lr_scheduler.step())
        else:
            trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)

    if to_save is not None:
        if output_path is None:
            raise ValueError("If to_save argument is provided then output_path argument should be also defined")
        checkpoint_handler = ModelCheckpoint(dirname=output_path, filename_prefix="training")
        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=save_every_iters), checkpoint_handler, to_save)

    if with_gpu_stats:
        GpuInfo().attach(trainer, name='gpu', event_name=Events.ITERATION_COMPLETED(every=log_every_iters))

    if output_names is not None:

        def output_transform(x, index, name):
            if isinstance(x, Mapping):
                return x[name]
            elif isinstance(x, Sequence):
                return x[index]
            elif isinstance(x, torch.Tensor):
                return x
            else:
                raise ValueError("Unhandled type of update_function's output. "
                                 "It should either mapping or sequence, but given {}".format(type(x)))

        for i, n in enumerate(output_names):
            RunningAverage(output_transform=partial(output_transform, index=i, name=n),
                           epoch_bound=False, device=device).attach(trainer, n)

    if with_pbars:
        if with_pbar_on_iters:
            ProgressBar(persist=False).attach(trainer, metric_names='all',
                                              event_name=Events.ITERATION_COMPLETED(every=log_every_iters))

        ProgressBar(persist=True, bar_format="").attach(trainer,
                                                        event_name=Events.EPOCH_STARTED,
                                                        closing_event_name=Events.COMPLETED)


def _setup_common_distrib_training_handlers(trainer, train_sampler=None,
                                            to_save=None, save_every_iters=1000, output_path=None,
                                            lr_scheduler=None, with_gpu_stats=True,
                                            output_names=None, with_pbars=True, with_pbar_on_iters=True,
                                            log_every_iters=100, device='cuda'):
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("Distributed setting is not initialized, please call `dist.init_process_group` before.")

    _setup_common_training_handlers(trainer, to_save=None,
                                    lr_scheduler=lr_scheduler, with_gpu_stats=with_gpu_stats,
                                    output_names=output_names,
                                    with_pbars=(dist.get_rank() == 0) and with_pbars,
                                    with_pbar_on_iters=with_pbar_on_iters,
                                    log_every_iters=log_every_iters, device=device)

    if train_sampler is not None:
        if not callable(getattr(train_sampler, "set_epoch", None)):
            raise TypeError("Train sampler should have `set_epoch` method")

        @trainer.on(Events.EPOCH_STARTED)
        def distrib_set_epoch(engine):
            train_sampler.set_epoch(engine.state.epoch - 1)

    if dist.get_rank() == 0:
        if to_save is not None:
            if output_path is None:
                raise ValueError("If to_save argument is provided then output_path argument should be also defined")
            checkpoint_handler = ModelCheckpoint(dirname=output_path, filename_prefix="training")
            trainer.add_event_handler(Events.ITERATION_COMPLETED(every=save_every_iters), checkpoint_handler, to_save)

    return trainer


def empty_cuda_cache(_):
    torch.cuda.empty_cache()
    import gc
    gc.collect()


def setup_any_logging(logger, logger_module, trainer, optimizers, evaluators, log_every_iters):
    if optimizers is not None:
        from torch.optim.optimizer import Optimizer

        if not isinstance(optimizers, (Optimizer, Mapping)):
            raise TypeError("Argument optimizers should be either a single optimizer or a dictionary or optimizers")

    if evaluators is not None:
        if not isinstance(evaluators, (Engine, Mapping)):
            raise TypeError("Argument optimizers should be either a single optimizer or a dictionary or optimizers")

    if log_every_iters is None:
        log_every_iters = 1

    logger.attach(trainer,
                  log_handler=logger_module.OutputHandler(tag="training", metric_names='all'),
                  event_name=Events.ITERATION_COMPLETED(every=log_every_iters))

    if optimizers is not None:
        # Log optimizer parameters
        if isinstance(optimizers, Optimizer):
            optimizers = {None: optimizers}

        for k, optimizer in optimizers.items():
            logger.attach(trainer,
                          log_handler=logger_module.OptimizerParamsHandler(optimizer, param_name="lr", tag=k),
                          event_name=Events.ITERATION_STARTED(every=log_every_iters))

    if evaluators is not None:
        # Log evaluation metrics
        if isinstance(evaluators, Engine):
            evaluators = {"validation": evaluators}

        for k, evaluator in evaluators.items():
            gst = global_step_from_engine(trainer)
            logger.attach(evaluator,
                          log_handler=logger_module.OutputHandler(tag=k, metric_names='all', global_step_transform=gst),
                          event_name=Events.COMPLETED)


def setup_tb_logging(output_path, trainer, optimizers=None, evaluators=None, log_every_iters=100):
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

    Returns:
        TensorboardLogger
    """
    tb_logger = TensorboardLogger(log_dir=output_path)
    setup_any_logging(tb_logger, tb_logger_module,
                      trainer, optimizers, evaluators,
                      log_every_iters=log_every_iters)
    return tb_logger


def setup_mlflow_logging(trainer, optimizers=None, evaluators=None, log_every_iters=100):
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

    Returns:
        MLflowLogger
    """
    mlflow_logger = MLflowLogger()
    setup_any_logging(mlflow_logger, mlflow_logger_module,
                      trainer, optimizers, evaluators,
                      log_every_iters=log_every_iters)
    return mlflow_logger


def setup_plx_logging(trainer, optimizers=None, evaluators=None, log_every_iters=100):
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

    Returns:
        PolyaxonLogger
    """
    plx_logger = PolyaxonLogger()
    setup_any_logging(plx_logger, polyaxon_logger_module,
                      trainer, optimizers, evaluators,
                      log_every_iters=log_every_iters)
    return plx_logger


def get_default_score_fn(metric_name):
    def wrapper(engine):
        score = engine.state.metrics[metric_name]
        return score

    return wrapper


def save_best_model_by_val_score(output_path, evaluator, model, metric_name, n_saved=3, trainer=None, tag="val"):
    """Method adds a handler to `evaluator` to save best models based on the score (named by `metric_name`)
    provided by `evaluator`.

    Args:
        output_path (str): output path to indicate where to save best models
        evaluator (Engine): evaluation engine used to provide the score
        model (nn.Module): model to store
        metric_name (str): metric name to use for score evaluation. This metric should be present in
            `evaluator.state.metrics`.
        n_saved (int, optional): number of best models to store
        trainer (Engine, optional): trainer engine to fetch the epoch when saving the best model.
        tag (str, optional): score name prefix: `{tag}_{metric_name}`. By default, tag is "val".

    """
    global_step_transform = None
    if trainer is not None:
        global_step_transform = global_step_from_engine(trainer)

    best_model_handler = ModelCheckpoint(dirname=output_path,
                                         filename_prefix="best",
                                         n_saved=n_saved,
                                         global_step_transform=global_step_transform,
                                         score_name="{}_{}".format(tag, metric_name.lower()),
                                         score_function=get_default_score_fn(metric_name))
    evaluator.add_event_handler(Events.COMPLETED, best_model_handler, {'model': model, })


def add_early_stopping_by_val_score(patience, evaluator, trainer, metric_name):
    """Method setups early stopping handler based on the score (named by `metric_name`) provided by `evaluator`.

    Args:
        patience (int): number of events to wait if no improvement and then stop the training.
        evaluator (Engine): evaluation engine used to provide the score
        trainer (Engine): trainer engine to stop the run if no improvement.
        metric_name (str): metric name to use for score evaluation. This metric should be present in
            `evaluator.state.metrics`.

    """
    es_handler = EarlyStopping(patience=patience, score_function=get_default_score_fn(metric_name), trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, es_handler)
