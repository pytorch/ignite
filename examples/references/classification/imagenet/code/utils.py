from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from ignite.engine import Events, Engine
from ignite.handlers import EarlyStopping, ModelCheckpoint, TerminateOnNan
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OutputHandler as tbOutputHandler, \
    OptimizerParamsHandler as tbOptimizerParamsHandler
from ignite.metrics import RunningAverage

from dataflow import setup_distrib_sampler

from mlflow_logger import MLflowLogger, OutputHandler as mfOutputHandler, \
    OptimizerParamsHandler as mfOptimizerParamsHandler


def initialize_amp(model, optimizer, fp16_opt_level):
    model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level, num_losses=1)
    model = DDP(model, delay_allreduce=True)
    return model, optimizer


def setup_distrib_loader(mode, config):
    assert mode in ("train", "val", "train_eval")
    loader = getattr(config, "{}_loader".format(mode), None)
    assert loader is not None, "Config should define '{}_loader'".format(mode)

    if isinstance(loader, DataLoader):
        if isinstance(loader.sampler, DistributedSampler):
            return loader, loader.sampler
        else:
            loader, sampler = setup_distrib_sampler(loader)
            return loader, sampler
    else:
        sampler = getattr(config, "{}_sampler".format(mode), None)
        assert sampler is not None and isinstance(sampler, DistributedSampler), \
            "As provided data loader is not torch.utils.data.DataLoader, config should " \
            "provide '{}_sampler' as DistributedSampler".format(mode)
        return loader, sampler


def setup_distrib_trainer(model, optimizer, criterion, train_sampler, device, config):

    prepare_batch = config.prepare_batch
    non_blocking = config.non_blocking
    accumulation_steps = getattr(config, "accumulation_steps", 1)

    def train_update_function(engine, batch):

        model.train()

        if engine.state.iteration % accumulation_steps == 0:
            optimizer.zero_grad()

        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)

        loss = criterion(y_pred, y) / accumulation_steps

        with amp.scale_loss(loss, optimizer, loss_id=0) as scaled_loss:
            scaled_loss.backward()

        if engine.state.iteration % accumulation_steps == 0:
            optimizer.step()

        return {
            'supervised batch loss': loss.item(),
        }

    trainer = Engine(train_update_function)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    @trainer.on(Events.EPOCH_STARTED)
    def distrib_set_epoch(engine):
        train_sampler.set_epoch(engine.state.epoch - 1)

    lr_scheduler = config.lr_scheduler
    if isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
        trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda engine: lr_scheduler.step())
    else:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, config.lr_scheduler)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)

    if dist.get_rank() == 0:
        # Checkpoint training
        checkpoint_handler = ModelCheckpoint(dirname=config.output_path.as_posix(),
                                             filename_prefix="checkpoint",
                                             save_interval=1000)
        trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                  checkpoint_handler,
                                  {'model': model, 'optimizer': optimizer})

        # Logging training with TQDM
        metric_names = [
            'supervised batch loss',
        ]

        def output_transform(x, name):
            return x[name]

        for n in metric_names:
            RunningAverage(output_transform=partial(output_transform, name=n), epoch_bound=False).attach(trainer, n)

        ProgressBar(persist=False).attach(trainer, metric_names)
        ProgressBar(persist=True, bar_format="").attach(trainer,
                                                        event_name=Events.EPOCH_STARTED,
                                                        closing_event_name=Events.COMPLETED)
    return trainer


def setup_distrib_evaluators(model, device, val_metrics, config):

    prepare_batch = config.prepare_batch
    non_blocking = config.non_blocking

    def eval_update_function(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            return {
                "y_pred": y_pred,
                "y": y
            }

    train_evaluator = Engine(eval_update_function)
    evaluator = Engine(eval_update_function)

    for name, metric in val_metrics.items():
        metric.attach(evaluator, name)
        metric.attach(train_evaluator, name)

    if dist.get_rank() == 0:
        ProgressBar(persist=False, desc="Train Evaluation").attach(train_evaluator)
        ProgressBar(persist=False, desc="Val Evaluation").attach(evaluator)

    evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)
    train_evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)

    return train_evaluator, evaluator


def setup_tb_mlflow_logging(trainer, optimizer, train_evaluator, evaluator, config):

    tb_logger = TensorboardLogger(log_dir=config.output_path.as_posix())
    mlflow_logger = MLflowLogger()

    for logger, handler in zip([tb_logger, mlflow_logger], [tbOutputHandler, mfOutputHandler]):
        logger.attach(trainer,
                      log_handler=handler(tag="training", metric_names='all'),
                      event_name=Events.ITERATION_COMPLETED)

    # Log optimizer parameters
    for logger, handler in zip([tb_logger, mlflow_logger], [tbOptimizerParamsHandler, mfOptimizerParamsHandler]):
        logger.attach(trainer,
                      log_handler=handler(optimizer, param_name="lr"),
                      event_name=Events.ITERATION_STARTED)

    for logger, handler in zip([tb_logger, mlflow_logger], [tbOutputHandler, mfOutputHandler]):
        # Log train eval metrics:
        logger.attach(train_evaluator,
                      log_handler=handler(tag="training",
                                          metric_names='all',
                                          another_engine=trainer),
                      event_name=Events.COMPLETED)

        # Log val metrics:
        logger.attach(evaluator,
                      log_handler=handler(tag="validation",
                                          metric_names='all',
                                          another_engine=trainer),
                      event_name=Events.COMPLETED)


def get_default_score_fn(metric_name):

    def wrapper(engine):
        score = engine.state.metrics[metric_name]
        return score

    return wrapper


def save_best_model_by_val_score(evaluator, model, metric_name, config):

    score_function = getattr(config, "score_function", get_default_score_fn(metric_name))

    best_model_handler = ModelCheckpoint(dirname=config.output_path.as_posix(),
                                         filename_prefix="best",
                                         n_saved=3,
                                         score_name="val_{}".format(metric_name.lower()),
                                         score_function=score_function)
    evaluator.add_event_handler(Events.COMPLETED, best_model_handler, {'model': model, })


def add_early_stopping_by_val_score(evaluator, trainer, metric_name, config):

    score_function = getattr(config, "score_function", get_default_score_fn(metric_name))
    es_handler = EarlyStopping(patience=config.es_patience, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, es_handler)


def empty_cuda_cache(engine):
    torch.cuda.empty_cache()
    import gc
    gc.collect()
