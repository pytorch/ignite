# This a training script launched with py_config_runner
# It should obligatory contain `run(config, **kwargs)` method

from pathlib import Path

import torch
from apex import amp
from py_config_runner.config_utils import assert_config, get_params, TRAINVAL_CONFIG
from py_config_runner.utils import set_seed
from utils import exp_tracking
from utils.handlers import predictions_gt_images_handler

import ignite
import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.engine import _prepare_batch, create_supervised_evaluator, Engine, Events
from ignite.metrics import Accuracy, TopKCategoricalAccuracy
from ignite.utils import setup_logger


def initialize(config):

    model = config.model.to(config.device)
    optimizer = config.optimizer
    # Setup Nvidia/Apex AMP
    model, optimizer = amp.initialize(model, optimizer, opt_level=getattr(config, "fp16_opt_level", "O2"), num_losses=1)

    # Adapt model to dist conf
    model = idist.auto_model(model)

    criterion = config.criterion.to(config.device)

    return model, optimizer, criterion


def create_trainer(model, optimizer, criterion, train_sampler, config, logger):
    prepare_batch = config.prepare_batch
    device = config.device

    # Setup trainer
    accumulation_steps = getattr(config, "accumulation_steps", 1)
    model_output_transform = getattr(config, "model_output_transform", lambda x: x)

    def train_update_function(engine, batch):

        model.train()

        x, y = prepare_batch(batch, device=device, non_blocking=True)
        y_pred = model(x)
        y_pred = model_output_transform(y_pred)
        loss = criterion(y_pred, y) / accumulation_steps

        with amp.scale_loss(loss, optimizer, loss_id=0) as scaled_loss:
            scaled_loss.backward()

        if engine.state.iteration % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return {
            "supervised batch loss": loss.item(),
        }

    output_names = getattr(config, "output_names", ["supervised batch loss"])
    lr_scheduler = config.lr_scheduler

    trainer = Engine(train_update_function)
    trainer.logger = logger

    to_save = {"model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler, "trainer": trainer, "amp": amp}

    save_every_iters = getattr(config, "save_every_iters", 1000)

    common.setup_common_training_handlers(
        trainer,
        train_sampler,
        to_save=to_save,
        save_every_iters=save_every_iters,
        output_path=config.output_path.as_posix(),
        lr_scheduler=lr_scheduler,
        with_gpu_stats=True,
        output_names=output_names,
        with_pbars=False,
    )

    common.ProgressBar(persist=False).attach(trainer, metric_names="all")

    return trainer


def create_evaluators(model, metrics, config):
    model_output_transform = getattr(config, "model_output_transform", lambda x: x)

    evaluator_args = dict(
        model=model,
        metrics=metrics,
        device=config.device,
        non_blocking=True,
        prepare_batch=config.prepare_batch,
        output_transform=lambda x, y, y_pred: (model_output_transform(y_pred), y),
    )
    train_evaluator = create_supervised_evaluator(**evaluator_args)
    evaluator = create_supervised_evaluator(**evaluator_args)

    common.ProgressBar(persist=False).attach(train_evaluator)
    common.ProgressBar(persist=False).attach(evaluator)

    return evaluator, train_evaluator


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed} - {tag} metrics:\n {metrics_output}")


def log_basic_info(logger, config):

    msg = f"\n- PyTorch version: {torch.__version__}"
    msg += f"\n- Ignite version: {ignite.__version__}"
    logger.info(msg)

    if idist.get_world_size() > 1:
        msg = "\nDistributed setting:"
        msg += f"\tbackend: {idist.backend()}"
        msg += f"\trank: {idist.get_rank()}"
        msg += f"\tworld size: {idist.get_world_size()}"
        logger.info(msg)


def training(local_rank, config, logger=None):

    if not getattr(config, "use_fp16", True):
        raise RuntimeError("This training script uses by default fp16 AMP")

    torch.backends.cudnn.benchmark = True

    set_seed(config.seed + local_rank)

    train_loader, val_loader, train_eval_loader = config.train_loader, config.val_loader, config.train_eval_loader

    # Setup model, optimizer, criterion
    model, optimizer, criterion = initialize(config)

    if not hasattr(config, "prepare_batch"):
        config.prepare_batch = _prepare_batch

    # Setup trainer for this specific task
    trainer = create_trainer(model, optimizer, criterion, train_loader.sampler, config, logger)

    if getattr(config, "benchmark_dataflow", False):
        benchmark_dataflow_num_iters = getattr(config, "benchmark_dataflow_num_iters", 1000)
        DataflowBenchmark(benchmark_dataflow_num_iters, prepare_batch=config.prepare_batch).attach(
            trainer, train_loader
        )

    # Setup evaluators
    val_metrics = {
        "Accuracy": Accuracy(),
        "Top-5 Accuracy": TopKCategoricalAccuracy(k=5),
    }

    if hasattr(config, "val_metrics") and isinstance(config.val_metrics, dict):
        val_metrics.update(config.val_metrics)

    evaluator, train_evaluator = create_evaluators(model, val_metrics, config)

    @trainer.on(Events.EPOCH_COMPLETED(every=getattr(config, "val_interval", 1)) | Events.COMPLETED)
    def run_validation():
        epoch = trainer.state.epoch
        state = train_evaluator.run(train_eval_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Train", state.metrics)
        state = evaluator.run(val_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Test", state.metrics)

    if getattr(config, "start_by_validation", False):
        trainer.add_event_handler(Events.STARTED, run_validation)

    score_metric_name = "Accuracy"

    if hasattr(config, "es_patience"):
        common.add_early_stopping_by_val_score(config.es_patience, evaluator, trainer, metric_name=score_metric_name)

    # Store 3 best models by validation accuracy:
    common.save_best_model_by_val_score(
        config.output_path.as_posix(),
        evaluator,
        model=model,
        metric_name=score_metric_name,
        n_saved=3,
        trainer=trainer,
        tag="val",
    )

    if idist.get_rank() == 0:

        tb_logger = common.setup_tb_logging(
            config.output_path.as_posix(),
            trainer,
            optimizer,
            evaluators={"training": train_evaluator, "validation": evaluator},
        )

        exp_tracking_logger = exp_tracking.setup_logging(
            trainer, optimizer, evaluators={"training": train_evaluator, "validation": evaluator}
        )

        # Log train/val predictions:
        tb_logger.attach(
            evaluator,
            log_handler=predictions_gt_images_handler(
                img_denormalize_fn=config.img_denormalize, n_images=15, another_engine=trainer, prefix_tag="validation"
            ),
            event_name=Events.ITERATION_COMPLETED(once=len(val_loader) // 2),
        )

        tb_logger.attach(
            train_evaluator,
            log_handler=predictions_gt_images_handler(
                img_denormalize_fn=config.img_denormalize, n_images=15, another_engine=trainer, prefix_tag="training"
            ),
            event_name=Events.ITERATION_COMPLETED(once=len(train_eval_loader) // 2),
        )

    trainer.run(train_loader, max_epochs=config.num_epochs)

    if idist.get_rank() == 0:
        tb_logger.close()
        exp_tracking_logger.close()


def run(config, **kwargs):
    """This is the main method to run the training. As this training script is launched with `py_config_runner`
    it should obligatory contain `run(config, **kwargs)` method.

    """

    assert torch.cuda.is_available(), torch.cuda.is_available()
    assert torch.backends.cudnn.enabled, "Nvidia/Amp requires cudnn backend to be enabled."

    with idist.Parallel(backend="nccl") as parallel:

        logger = setup_logger(name="ImageNet Training", distributed_rank=idist.get_rank())

        assert_config(config, TRAINVAL_CONFIG)
        # The following attributes are automatically added by py_config_runner
        assert hasattr(config, "config_filepath") and isinstance(config.config_filepath, Path)
        assert hasattr(config, "script_filepath") and isinstance(config.script_filepath, Path)

        if idist.get_rank() == 0 and exp_tracking.has_clearml:
            try:
                from clearml import Task
            except ImportError:
                # Backwards-compatibility for legacy Trains SDK
                from trains import Task

            task = Task.init("ImageNet Training", config.config_filepath.stem)
            task.connect_configuration(config.config_filepath.as_posix())

        log_basic_info(logger, config)

        config.output_path = Path(exp_tracking.get_output_path())
        # dump python files to reproduce the run
        exp_tracking.log_artifact(config.config_filepath.as_posix())
        exp_tracking.log_artifact(config.script_filepath.as_posix())
        exp_tracking.log_inputs(get_params(config, TRAINVAL_CONFIG))

        try:
            parallel.run(training, config, logger=logger)
        except KeyboardInterrupt:
            logger.info("Catched KeyboardInterrupt -> exit")
        except Exception as e:  # noqa
            logger.exception("")
            raise e


class DataflowBenchmark:
    def __init__(self, num_iters=100, prepare_batch=None):

        from ignite.handlers import Timer

        device = idist.device()

        def upload_to_gpu(engine, batch):
            if prepare_batch is not None:
                x, y = prepare_batch(batch, device=device, non_blocking=False)

        self.num_iters = num_iters
        self.benchmark_dataflow = Engine(upload_to_gpu)

        @self.benchmark_dataflow.on(Events.ITERATION_COMPLETED(once=num_iters))
        def stop_benchmark_dataflow(engine):
            engine.terminate()

        if idist.get_rank() == 0:

            @self.benchmark_dataflow.on(Events.ITERATION_COMPLETED(every=num_iters // 100))
            def show_progress_benchmark_dataflow(engine):
                print(".", end=" ")

        self.timer = Timer(average=False)
        self.timer.attach(
            self.benchmark_dataflow,
            start=Events.EPOCH_STARTED,
            resume=Events.ITERATION_STARTED,
            pause=Events.ITERATION_COMPLETED,
            step=Events.ITERATION_COMPLETED,
        )

    def attach(self, trainer, train_loader):

        from torch.utils.data import DataLoader

        @trainer.on(Events.STARTED)
        def run_benchmark(_):
            if idist.get_rank() == 0:
                print("-" * 50)
                print(" - Dataflow benchmark")

            self.benchmark_dataflow.run(train_loader)
            t = self.timer.value()

            if idist.get_rank() == 0:
                print(" ")
                print(f" Total time ({self.num_iters} iterations) : {t:.5f} seconds")
                print(f" time per iteration         : {t / self.num_iters} seconds")

                if isinstance(train_loader, DataLoader):
                    num_images = train_loader.batch_size * self.num_iters
                    print(f" number of images / s       : {num_images / t}")

                print("-" * 50)
