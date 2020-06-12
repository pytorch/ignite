# This a training script launched with py_config_runner
# It should obligatory contain `run(config, **kwargs)` method

from collections.abc import Mapping

import torch
import torch.nn as nn

from apex import amp

import ignite.distributed as idist
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import ConfusionMatrix, IoU, mIoU

from ignite.contrib.engines import common

from py_config_runner.utils import set_seed

from utils.handlers import predictions_gt_images_handler


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

    common.setup_common_training_handlers(
        trainer,
        train_sampler,
        to_save=to_save,
        save_every_iters=1000,
        output_path=config.output_path.as_posix(),
        lr_scheduler=config.lr_scheduler,
        with_gpu_stats=True,
        output_names=output_names,
        with_pbars=False,
    )

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

    return evaluator, train_evaluator


def training(local_rank, config, with_mlflow_logging=False, with_plx_logging=False, with_trains_logging=False):

    if not getattr(config, "use_fp16", True):
        raise RuntimeError("This training script uses by default fp16 AMP")

    torch.backends.cudnn.benchmark = True

    set_seed(config.seed + local_rank)
    device = config.device

    train_loader, val_loader, train_eval_loader = config.train_loader, config.val_loader, config.train_eval_loader

    # Setup model, optimizer, criterion
    model, optimizer, criterion = initialize(config)

    # Setup trainer for this specific task
    trainer = create_trainer(model, optimizer, criterion, config)

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

    def run_validation(_):
        train_evaluator.run(train_eval_loader)
        evaluator.run(val_loader)

    if getattr(config, "start_by_validation", False):
        trainer.add_event_handler(Events.STARTED, run_validation)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=getattr(config, "val_interval", 1)), run_validation)
    trainer.add_event_handler(Events.COMPLETED, run_validation)

    score_metric_name = "mIoU_bg"

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

        if with_mlflow_logging:
            common.setup_mlflow_logging(
                trainer, optimizer, evaluators={"training": train_evaluator, "validation": evaluator}
            )

        if with_plx_logging:
            common.setup_plx_logging(
                trainer, optimizer, evaluators={"training": train_evaluator, "validation": evaluator}
            )

        if with_trains_logging:
            common.setup_trains_logging(
                trainer, optimizer, evaluators={"training": train_evaluator, "validation": evaluator}
            )

        # Log val predictions:
        tb_logger.attach(
            evaluator,
            log_handler=predictions_gt_images_handler(
                img_denormalize_fn=config.img_denormalize, n_images=15, another_engine=trainer, prefix_tag="validation"
            ),
            event_name=Events.ITERATION_COMPLETED(once=len(val_loader) // 2),
        )

    trainer.run(train_loader, max_epochs=config.num_epochs)

    if idist.get_rank() == 0:
        tb_logger.close()
