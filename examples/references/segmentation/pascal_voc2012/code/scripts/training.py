# This a training script launched with py_config_runner
# It should obligatory contain `run(config, **kwargs)` method

from pathlib import Path
from collections.abc import Mapping

import torch

from apex import amp

import ignite
import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.handlers import DiskSaver
from ignite.metrics import ConfusionMatrix, IoU, mIoU
from ignite.utils import setup_logger

from py_config_runner.utils import set_seed
from py_config_runner.config_utils import get_params, TRAINVAL_CONFIG, assert_config

import sys

# Adds "code" folder to python path
sys.path.insert(0, Path(__file__).parent.parent.as_posix())

from utils.handlers import predictions_gt_images_handler
from utils import exp_tracking
from dataflow.datasets import VOCSegmentationOpencv


def initialize(config):

    model = config.model.to(config.device)
    optimizer = config.optimizer
    # Setup Nvidia/Apex AMP
    model, optimizer = amp.initialize(model, optimizer, opt_level=getattr(config, "fp16_opt_level", "O2"), num_losses=1)

    # Adapt model to dist conf
    model = idist.auto_model(model)

    criterion = config.criterion.to(config.device)

    return model, optimizer, criterion


def get_save_handler(config):
    if exp_tracking.has_trains:
        from ignite.contrib.handlers.trains_logger import TrainsSaver

        return TrainsSaver(dirname=config.output_path.as_posix())

    return DiskSaver(config.output_path.as_posix())


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
        loss = criterion(y_pred, y)

        if isinstance(loss, Mapping):
            assert "supervised batch loss" in loss
            loss_dict = loss
            output = {k: v.item() for k, v in loss_dict.items()}
            loss = loss_dict["supervised batch loss"] / accumulation_steps
        else:
            output = {"supervised batch loss": loss.item()}

        with amp.scale_loss(loss, optimizer, loss_id=0) as scaled_loss:
            scaled_loss.backward()

        if engine.state.iteration % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return output

    output_names = getattr(config, "output_names", ["supervised batch loss",])
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
        save_handler=get_save_handler(config),
        lr_scheduler=lr_scheduler,
        with_gpu_stats=exp_tracking.has_mlflow,
        output_names=output_names,
        with_pbars=False,
    )

    if idist.get_rank() == 0:
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
        output_transform=lambda x, y, y_pred: (model_output_transform(y_pred), y,),
    )
    train_evaluator = create_supervised_evaluator(**evaluator_args)
    evaluator = create_supervised_evaluator(**evaluator_args)

    if idist.get_rank() == 0:
        common.ProgressBar(desc="Evaluation (train)", persist=False).attach(train_evaluator)
        common.ProgressBar(desc="Evaluation (val)", persist=False).attach(evaluator)

    return evaluator, train_evaluator


def log_metrics(logger, epoch, elapsed, tag, metrics):
    logger.info(
        "\nEpoch {} - Evaluation time (seconds): {} - {} metrics:\n {}".format(
            epoch, int(elapsed), tag, "\n".join(["\t{}: {}".format(k, v) for k, v in metrics.items()])
        )
    )


def log_basic_info(logger, config):

    msg = "\n- PyTorch version: {}".format(torch.__version__)
    msg += "\n- Ignite version: {}".format(ignite.__version__)
    msg += "\n- Cuda device name: {}".format(torch.cuda.get_device_name(idist.get_local_rank()))

    logger.info(msg)

    if idist.get_world_size() > 1:
        msg = "\nDistributed setting:"
        msg += "\tbackend: {}".format(idist.backend())
        msg += "\trank: {}".format(idist.get_rank())
        msg += "\tworld size: {}".format(idist.get_world_size())
        logger.info(msg)


def training(local_rank, config, logger=None):

    if not getattr(config, "use_fp16", True):
        raise RuntimeError("This training script uses by default fp16 AMP")

    torch.backends.cudnn.benchmark = True

    set_seed(config.seed + local_rank)

    train_loader, val_loader, train_eval_loader = config.train_loader, config.val_loader, config.train_eval_loader

    # Setup model, optimizer, criterion
    model, optimizer, criterion = initialize(config)

    # Setup trainer for this specific task
    trainer = create_trainer(model, optimizer, criterion, train_loader.sampler, config, logger)

    # Setup evaluators
    num_classes = config.num_classes
    cm_metric = ConfusionMatrix(num_classes=num_classes)

    val_metrics = {
        "IoU": IoU(cm_metric),
        "mIoU_bg": mIoU(cm_metric),
    }

    if hasattr(config, "val_metrics") and isinstance(config.val_metrics, dict):
        val_metrics.update(config.val_metrics)

    evaluator, train_evaluator = create_evaluators(model, val_metrics, config)

    val_interval = getattr(config, "val_interval", 1)

    @trainer.on(Events.EPOCH_COMPLETED(every=val_interval))
    def run_validation():
        epoch = trainer.state.epoch
        state = train_evaluator.run(train_eval_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Train", state.metrics)
        state = evaluator.run(val_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Test", state.metrics)

    if config.num_epochs % val_interval != 0:
        trainer.add_event_handler(Events.COMPLETED, run_validation)

    if getattr(config, "start_by_validation", False):
        trainer.add_event_handler(Events.STARTED, run_validation)

    score_metric_name = "mIoU_bg"

    if hasattr(config, "es_patience"):
        common.add_early_stopping_by_val_score(config.es_patience, evaluator, trainer, metric_name=score_metric_name)

    # Store 3 best models by validation accuracy:
    common.gen_save_best_models_by_val_score(
        save_handler=get_save_handler(config),
        evaluator=evaluator,
        models=model,
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

        if not exp_tracking.has_trains:
            exp_tracking_logger = exp_tracking.setup_logging(
                trainer, optimizer, evaluators={"training": train_evaluator, "validation": evaluator}
            )

        # Log validation predictions as images
        # We define a custom event filter to log less frequently the images (to reduce storage size)
        # - we plot images with masks of the middle validation batch
        # - once every 3 validations and
        # - at the end of the training
        def custom_event_filter(_, val_iteration):
            c1 = val_iteration == len(val_loader) // 2
            c2 = trainer.state.epoch % (getattr(config, "val_interval", 1) * 3) == 0
            c2 |= trainer.state.epoch == config.num_epochs
            return c1 and c2

        tb_logger.attach(
            evaluator,
            log_handler=predictions_gt_images_handler(
                img_denormalize_fn=config.img_denormalize, n_images=15, another_engine=trainer, prefix_tag="validation"
            ),
            event_name=Events.ITERATION_COMPLETED(event_filter=custom_event_filter),
        )

    # Log confusion matrix to Trains:
    if exp_tracking.has_trains:

        @trainer.on(Events.COMPLETED)
        def compute_and_log_cm():
            cm = cm_metric.compute()
            # CM: values are normalized such that diagonal values represent class recalls
            cm = ConfusionMatrix.normalize(cm, "recall").cpu().numpy()

            if idist.get_rank() == 0:
                from trains import Task

                trains_logger = Task.current_task().get_logger()
                trains_logger.report_confusion_matrix(
                    title="Final Confusion Matrix",
                    series="cm-preds-gt",
                    matrix=cm,
                    iteration=trainer.state.iteration,
                    xlabels=VOCSegmentationOpencv.target_names,
                    ylabels=VOCSegmentationOpencv.target_names,
                )

    trainer.run(train_loader, max_epochs=config.num_epochs)

    if idist.get_rank() == 0:
        tb_logger.close()
        if not exp_tracking.has_trains:
            exp_tracking_logger.close()


def run(config, **kwargs):
    """This is the main method to run the training. As this training script is launched with `py_config_runner`
    it should obligatory contain `run(config, **kwargs)` method.

    """

    assert torch.cuda.is_available(), torch.cuda.is_available()
    assert torch.backends.cudnn.enabled, "Nvidia/Amp requires cudnn backend to be enabled."

    with idist.Parallel(backend="nccl") as parallel:

        logger = setup_logger(name="Pascal-VOC12 Training", distributed_rank=idist.get_rank())

        assert_config(config, TRAINVAL_CONFIG)
        # The following attributes are automatically added by py_config_runner
        assert hasattr(config, "config_filepath") and isinstance(config.config_filepath, Path)
        assert hasattr(config, "script_filepath") and isinstance(config.script_filepath, Path)

        if idist.get_rank() == 0 and exp_tracking.has_trains:
            from trains import Task

            task = Task.init("Pascal-VOC12 Training", config.config_filepath.stem)
            task.connect_configuration(config.config_filepath.as_posix())

        log_basic_info(logger, config)

        config.output_path = Path(exp_tracking.get_output_path())
        # dump python files to reproduce the run
        exp_tracking.log_artifact(config.config_filepath.as_posix())
        exp_tracking.log_artifact(config.script_filepath.as_posix())
        exp_tracking.log_params(get_params(config, TRAINVAL_CONFIG))

        try:
            parallel.run(training, config, logger=logger)
        except KeyboardInterrupt:
            logger.info("Catched KeyboardInterrupt -> exit")
        except Exception as e:  # noqa
            logger.exception("")
            raise e
