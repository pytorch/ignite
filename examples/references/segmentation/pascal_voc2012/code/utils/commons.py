from functools import partial

import torch
import torch.distributed as dist

from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from ignite.engine import Events, Engine
from ignite.metrics import RunningAverage
from ignite.handlers import EarlyStopping, ModelCheckpoint, TerminateOnNan
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers import TensorboardLogger
import ignite.contrib.handlers.tensorboard_logger as tb_logger_module
from ignite.contrib.handlers import MLflowLogger
import ignite.contrib.handlers.mlflow_logger as mlflow_logger_module
from ignite.contrib.handlers import PolyaxonLogger
import ignite.contrib.handlers.polyaxon_logger as polyaxon_logger_module
from ignite.contrib.metrics.gpu_info import GpuInfo


def initialize_amp(model, optimizer, fp16_opt_level):
    model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level, num_losses=1)
    if isinstance(model, list):
        for i, m in enumerate(model):
            model[i] = DDP(m, delay_allreduce=True)
    return model, optimizer


def setup_distrib_trainer(train_update_function, model, optimizer, train_sampler, config, setup_pbar_on_iters=True):

    trainer = Engine(train_update_function)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    @trainer.on(Events.EPOCH_STARTED)
    def distrib_set_epoch(engine):
        train_sampler.set_epoch(engine.state.epoch - 1)

    lr_scheduler = config.lr_scheduler
    if isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
        trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda engine: lr_scheduler.step())
    else:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, lr_scheduler)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)

    GpuInfo().attach(trainer, name='gpu')

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

        if setup_pbar_on_iters:
            ProgressBar(persist=False).attach(trainer, metric_names)

        ProgressBar(persist=True, bar_format="").attach(trainer,
                                                        event_name=Events.EPOCH_STARTED,
                                                        closing_event_name=Events.COMPLETED)
    return trainer


def setup_distrib_evaluators(model, device, val_metrics, config):

    prepare_batch = config.prepare_batch
    non_blocking = config.non_blocking
    model_output_transform = getattr(config, "model_output_transform", lambda x: x)

    def eval_update_function(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            y_pred = model_output_transform(y_pred)
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


def _setup_x_logging(logger, logger_module, trainer, optimizer, train_evaluator, evaluator):
    logger.attach(trainer,
                  log_handler=logger_module.OutputHandler(tag="training", metric_names='all'),
                  event_name=Events.ITERATION_COMPLETED)

    # Log optimizer parameters
    logger.attach(trainer,
                  log_handler=logger_module.OptimizerParamsHandler(optimizer, param_name="lr"),
                  event_name=Events.ITERATION_STARTED)

    logger.attach(train_evaluator,
                  log_handler=logger_module.OutputHandler(tag="training",
                                                          metric_names='all',
                                                          another_engine=trainer),
                  event_name=Events.COMPLETED)

    # Log val metrics:
    logger.attach(evaluator,
                  log_handler=logger_module.OutputHandler(tag="validation",
                                                          metric_names='all',
                                                          another_engine=trainer),
                  event_name=Events.COMPLETED)


def setup_tb_logging(trainer, optimizer, train_evaluator, evaluator, config):
    tb_logger = TensorboardLogger(log_dir=config.output_path.as_posix())
    _setup_x_logging(tb_logger, tb_logger_module, trainer, optimizer, train_evaluator, evaluator)
    return tb_logger


def setup_mlflow_logging(trainer, optimizer, train_evaluator, evaluator, **kwargs):
    mlflow_logger = MLflowLogger()
    _setup_x_logging(mlflow_logger, mlflow_logger_module, trainer, optimizer, train_evaluator, evaluator)
    return mlflow_logger


def setup_plx_logging(trainer, optimizer, train_evaluator, evaluator, **kwargs):
    plx_logger = PolyaxonLogger()
    _setup_x_logging(plx_logger, polyaxon_logger_module, trainer, optimizer, train_evaluator, evaluator)
    return plx_logger


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


def get_artifact_path(run_uuid, path):
    import mlflow
    client = mlflow.tracking.MlflowClient()
    return client.download_artifacts(run_uuid, path)
