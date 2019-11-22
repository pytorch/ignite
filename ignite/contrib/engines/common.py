from functools import partial

try:
    from collections.abc import Sequence, Mapping
except ImportError:  # Python 2.7 compatibility
    from collections import Sequence, Mapping

import torch
import torch.distributed as dist

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import TerminateOnNan, ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import GpuInfo


def create_common_trainer(update_function,
                          to_save=None, save_every=1000, output_path=None,
                          lr_scheduler=None, with_gpu_stats=True,
                          output_names=None, with_pbars=True, with_pbar_on_iters=True, log_every_iters=100):
    """Helper method to create a trainer with already attached several common handlers:
        - :class:`~ignite.handlers.TerminateOnNan`
        - handler to setup learning rate scheduling
        - :class:`~ignite.handlers.ModelCheckpoint`
        - :class:`~ignite.metrics.RunningAverage` on `update_function` output
        - Two progress bars on epochs and optionally on iterations

    Args:
        update_function (callable): trainer's update function. Output of `update_function` should be a dictionary
            or sequence or a single tensor.
        to_save (dict, optional): dictionary with objects to save in the checkpoint. This is used with
            :class:`~ignite.handlers.ModelCheckpoint`.
        save_every (int, optional): saving interval. By default, `to_save` objects are stored each 1000 iterations.
        output_path (str, optional): output path to indicate whether `to_save` objects are stored.
        lr_scheduler (ParamScheduler or subclass of `torch.optim.lr_scheduler._LRScheduler`): learning rate scheduler
            as native torch LRScheduler or ignite's parameter scheduler.
        with_gpu_stats (bool, optional): if True, :class:`~ignite.contrib.metrics.handlers.GpuInfo` is attached to the
            trainer
        output_names (list/tuple): list of names associated with `update_function` output dictionary.
        with_pbars (bool, optional): if True, two progress bars on epochs and optionally on iterations are attached
        with_pbar_on_iters (bool, optional): if True, a progress bar on interations is attached to the trainer.
        log_every_iters (int, optional): logging interval for :class:`~ignite.contrib.metrics.handlers.GpuInfo` and for
            epoch-wise progress bar.

    Returns:
        Engine
    """

    trainer = Engine(update_function)
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
        checkpoint_handler = ModelCheckpoint(dirname=output_path, filename_prefix="checkpoint", save_interval=1)
        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=save_every), checkpoint_handler, to_save)

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
                           epoch_bound=False).attach(trainer, n)

    if with_pbars:
        if with_pbar_on_iters:
            ProgressBar(persist=False).attach(trainer, metric_names='all',
                                              event_name=Events.ITERATION_COMPLETED(every=log_every_iters))

        ProgressBar(persist=True, bar_format="").attach(trainer,
                                                        event_name=Events.EPOCH_STARTED,
                                                        closing_event_name=Events.COMPLETED)

    return trainer


def create_common_distrib_trainer(update_function, train_sampler,
                                  to_save=None, save_every=1000, output_path=None,
                                  lr_scheduler=None, with_gpu_stats=True,
                                  output_names=None, with_pbars=True, with_pbar_on_iters=True, log_every_iters=100):
    """Helper method to create a trainer supporting data parallel distributed configuration with already attached
    several common handlers:
        - :class:`~ignite.handlers.TerminateOnNan`
        - handler to setup learning rate scheduling
        - :class:`~ignite.handlers.ModelCheckpoint`
        - :class:`~ignite.metrics.RunningAverage` on `update_function` output
        - Two progress bars on epochs and optionally on iterations

    Args:
        update_function (callable): trainer's update function. Output of `update_function` should be a dictionary
            or sequence or a single tensor.
        train_sampler (torch.utils.data.DistributedSampler): distributed sampler used to call `set_epoch` method on
            epoch started event.
        to_save (dict, optional): dictionary with objects to save in the checkpoint. This is used with
            :class:`~ignite.handlers.ModelCheckpoint`.
        save_every (int, optional): saving interval. By default, `to_save` objects are stored each 1000 iterations.
        output_path (str, optional): output path to indicate whether `to_save` objects are stored.
        lr_scheduler (ParamScheduler or subclass of `torch.optim.lr_scheduler._LRScheduler`): learning rate scheduler
            as native torch LRScheduler or ignite's parameter scheduler.
        with_gpu_stats (bool, optional): if True, :class:`~ignite.contrib.metrics.handlers.GpuInfo` is attached to the
            trainer
        output_names (list/tuple): list of names associated with `update_function` output.
        with_pbars (bool, optional): if True, two progress bars on epochs and optionally on iterations are attached
        with_pbar_on_iters (bool, optional): if True, a progress bar on interations is attached to the trainer.
        log_every_iters (int, optional): logging interval for :class:`~ignite.contrib.metrics.handlers.GpuInfo` and for
            epoch-wise progress bar.

    Returns:
        Engine
    """
    trainer = create_common_trainer(update_function, to_save=None,
                                    lr_scheduler=lr_scheduler,
                                    with_gpu_stats=with_gpu_stats,
                                    output_names=output_names,
                                    with_pbars=(dist.get_rank() == 0) and with_pbars,
                                    with_pbar_on_iters=with_pbar_on_iters,
                                    log_every_iters=log_every_iters)

    @trainer.on(Events.EPOCH_STARTED)
    def distrib_set_epoch(engine):
        train_sampler.set_epoch(engine.state.epoch - 1)

    if dist.get_rank() == 0:
        if to_save is not None:
            if output_path is None:
                raise ValueError("If to_save argument is provided then output_path argument should be also defined")
            checkpoint_handler = ModelCheckpoint(dirname=output_path, filename_prefix="checkpoint", save_interval=1)
            trainer.add_event_handler(Events.ITERATION_COMPLETED(every=save_every), checkpoint_handler, to_save)

    return trainer


def empty_cuda_cache(_):
    torch.cuda.empty_cache()
    import gc
    gc.collect()
