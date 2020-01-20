# This a training script launched with py_config_runner
# It should obligatory contain `run(config, **kwargs)` method

from collections.abc import Mapping

import torch
import torch.distributed as dist

from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from ignite.engine import Engine, Events, _prepare_batch, create_supervised_evaluator
from ignite.metrics import ConfusionMatrix, IoU, mIoU

from ignite.contrib.handlers import ProgressBar
from ignite.contrib.engines import common

from py_config_runner.utils import set_seed

from utils.handlers import predictions_gt_images_handler


def training(config, local_rank=None, with_mlflow_logging=False, with_plx_logging=False):

    if not getattr(config, "use_fp16", True):
        raise RuntimeError("This training script uses by default fp16 AMP")

    set_seed(config.seed + local_rank)
    torch.cuda.set_device(local_rank)
    device = 'cuda'

    torch.backends.cudnn.benchmark = True

    train_loader = config.train_loader
    train_sampler = getattr(train_loader, "sampler", None)
    assert train_sampler is not None, "Train loader of type '{}' " \
                                      "should have attribute 'sampler'".format(type(train_loader))
    assert hasattr(train_sampler, 'set_epoch') and callable(train_sampler.set_epoch), \
        "Train sampler should have a callable method `set_epoch`"

    train_eval_loader = config.train_eval_loader
    val_loader = config.val_loader

    model = config.model.to(device)
    optimizer = config.optimizer
    model, optimizer = amp.initialize(model, optimizer, opt_level=getattr(config, "fp16_opt_level", "O2"), num_losses=1)
    model = DDP(model, delay_allreduce=True)
    criterion = config.criterion.to(device)

    prepare_batch = getattr(config, "prepare_batch", _prepare_batch)
    non_blocking = getattr(config, "non_blocking", True)

    # Setup trainer
    accumulation_steps = getattr(config, "accumulation_steps", 1)
    model_output_transform = getattr(config, "model_output_transform", lambda x: x)

    def train_update_function(engine, batch):

        model.train()

        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        y_pred = model_output_transform(y_pred)
        loss = criterion(y_pred, y)

        if isinstance(loss, Mapping):
            assert 'supervised batch loss' in loss
            loss_dict = loss
            output = {k: v.item() for k, v in loss_dict.items()}
            loss = loss_dict['supervised batch loss'] / accumulation_steps
        else:
            output = {'supervised batch loss': loss.item()}

        with amp.scale_loss(loss, optimizer, loss_id=0) as scaled_loss:
            scaled_loss.backward()

        if engine.state.iteration % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return output

    output_names = getattr(config, "output_names", ['supervised batch loss', ])

    trainer = Engine(train_update_function)
    common.setup_common_distrib_training_handlers(
        trainer, train_sampler,
        to_save={'model': model, 'optimizer': optimizer},
        save_every_iters=1000, output_path=config.output_path.as_posix(),
        lr_scheduler=config.lr_scheduler, with_gpu_stats=True,
        output_names=output_names,
        with_pbars=True, with_pbar_on_iters=with_mlflow_logging,
        log_every_iters=1
    )

    # Setup evaluators
    num_classes = config.num_classes
    cm_metric = ConfusionMatrix(num_classes=num_classes)

    val_metrics = {
        "IoU": IoU(cm_metric),
        "mIoU_bg": mIoU(cm_metric),
    }

    if hasattr(config, "val_metrics") and isinstance(config.val_metrics, dict):
        val_metrics.update(config.val_metrics)

    model_output_transform = getattr(config, "model_output_transform", lambda x: x)

    evaluator_args = dict(
        model=model, metrics=val_metrics, device=device, non_blocking=non_blocking, prepare_batch=prepare_batch,
        output_transform=lambda x, y, y_pred: (model_output_transform(y_pred), y,)
    )
    train_evaluator = create_supervised_evaluator(**evaluator_args)
    evaluator = create_supervised_evaluator(**evaluator_args)

    if dist.get_rank() == 0 and with_mlflow_logging:
        ProgressBar(persist=False, desc="Train Evaluation").attach(train_evaluator)
        ProgressBar(persist=False, desc="Val Evaluation").attach(evaluator)

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

    if dist.get_rank() == 0:

        tb_logger = common.setup_tb_logging(config.output_path.as_posix(), trainer, optimizer,
                                            evaluators={"training": train_evaluator, "validation": evaluator})
        if with_mlflow_logging:
            common.setup_mlflow_logging(trainer, optimizer,
                                        evaluators={"training": train_evaluator, "validation": evaluator})

        if with_plx_logging:
            common.setup_plx_logging(trainer, optimizer,
                                     evaluators={"training": train_evaluator, "validation": evaluator})

        common.save_best_model_by_val_score(config.output_path.as_posix(), evaluator, model,
                                            metric_name=score_metric_name, trainer=trainer)

        # Log train/val predictions:
        tb_logger.attach(evaluator,
                         log_handler=predictions_gt_images_handler(img_denormalize_fn=config.img_denormalize,
                                                                   n_images=15,
                                                                   another_engine=trainer,
                                                                   prefix_tag="validation"),
                         event_name=Events.EPOCH_COMPLETED)

        log_train_predictions = getattr(config, "log_train_predictions", False)
        if log_train_predictions:
            tb_logger.attach(train_evaluator,
                             log_handler=predictions_gt_images_handler(img_denormalize_fn=config.img_denormalize,
                                                                       n_images=15,
                                                                       another_engine=trainer,
                                                                       prefix_tag="validation"),
                             event_name=Events.EPOCH_COMPLETED)

    trainer.run(train_loader, max_epochs=config.num_epochs)
