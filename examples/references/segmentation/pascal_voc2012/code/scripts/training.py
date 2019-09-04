# This a training script launched with py_config_runner
# It should obligatory contain `run(config, **kwargs)` method

from pathlib import Path

import torch
import torch.distributed as dist

from apex import amp

import mlflow

import ignite
from ignite.engine import Events, _prepare_batch
from ignite.metrics import ConfusionMatrix, IoU, mIoU

from py_config_runner.config_utils import get_params, TRAINVAL_CONFIG, assert_config
from py_config_runner.utils import set_seed

from tools.commons import initialize_amp, setup_distrib_trainer, setup_distrib_evaluators, setup_mlflow_logging, \
    setup_tb_logging, save_best_model_by_val_score, add_early_stopping_by_val_score, setup_distrib_loader

from tools.handlers import predictions_gt_images_handler


def training(config, local_rank=None):

    if not getattr(config, "use_fp16", True):
        raise RuntimeError("This training script uses by default fp16 AMP")

    set_seed(config.seed + local_rank)
    torch.cuda.set_device(local_rank)
    device = 'cuda'

    torch.backends.cudnn.benchmark = True

    if dist.get_rank() == 0:
        mlflow.log_params({
            "pytorch version": torch.__version__,
            "ignite version": ignite.__version__,
        })
        mlflow.log_params(get_params(config, TRAINVAL_CONFIG))

    train_loader, train_sampler = setup_distrib_loader("train", config)
    train_eval_loader, _ = setup_distrib_loader("train_eval", config)
    val_loader, _ = setup_distrib_loader("val", config)

    model = config.model.to(device)
    optimizer = config.optimizer
    model, optimizer = initialize_amp(model, optimizer, getattr(config, "fp16_opt_level", "O2"))
    criterion = config.criterion.to(device)

    # Setup trainer
    prepare_batch = getattr(config, "prepare_batch", _prepare_batch)
    non_blocking = getattr(config, "non_blocking", True)
    accumulation_steps = getattr(config, "accumulation_steps", 1)
    model_output_transform = getattr(config, "model_output_transform", lambda x: x)

    def train_update_function(engine, batch):

        model.train()

        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        y_pred = model_output_transform(y_pred)
        loss = criterion(y_pred, y) / accumulation_steps

        with amp.scale_loss(loss, optimizer, loss_id=0) as scaled_loss:
            scaled_loss.backward()

        if engine.state.iteration % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return {
            'supervised batch loss': loss.item(),
        }

    trainer = setup_distrib_trainer(train_update_function, model, optimizer, train_sampler, config)

    def output_transform(output):
        return output['y_pred'], output['y']

    num_classes = config.num_classes
    cm_metric = ConfusionMatrix(num_classes=num_classes, output_transform=output_transform)

    val_metrics = {
        "IoU": IoU(cm_metric),
        "mIoU_bg": mIoU(cm_metric),
    }

    if hasattr(config, "val_metrics") and isinstance(config.val_metrics, dict):
        val_metrics.update(config.val_metrics)

    train_evaluator, evaluator = setup_distrib_evaluators(model, device, val_metrics, config)

    val_interval = getattr(config, "val_interval", 1)
    start_by_validation = getattr(config, "start_by_validation", False)

    def run_validation(engine, val_interval):
        if engine.state.epoch > int(not start_by_validation) and ((engine.state.epoch - 1) % val_interval == 0):
            train_evaluator.run(train_eval_loader)
            evaluator.run(val_loader)

    trainer.add_event_handler(Events.EPOCH_STARTED, run_validation, val_interval=val_interval)
    trainer.add_event_handler(Events.COMPLETED, run_validation, val_interval=1)

    if dist.get_rank() == 0:

        tb_logger = setup_tb_logging(trainer, optimizer, train_evaluator, evaluator, config)
        setup_mlflow_logging(trainer, optimizer, train_evaluator, evaluator)

        save_best_model_by_val_score(evaluator, model, metric_name="mIoU_bg", config=config)

        if hasattr(config, "es_patience"):
            add_early_stopping_by_val_score(evaluator, trainer, metric_name="mIoU_bg", config=config)

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


def run(config, logger=None, local_rank=0, **kwargs):

    assert torch.cuda.is_available()
    assert torch.backends.cudnn.enabled, "Nvidia/Amp requires cudnn backend to be enabled."

    dist.init_process_group(getattr(config, "dist_backend", "nccl"),
                            init_method=getattr(config, "dist_url", "env://"))

    # We pass config with option --manual_config_load
    assert hasattr(config, "setup"), "We need to manually setup the configuration, please set --manual_config_load " \
                                     "to py_config_runner"

    config = config.setup()

    assert_config(config, TRAINVAL_CONFIG)
    # The following attributes are automatically added by py_config_runner
    assert hasattr(config, "config_filepath") and isinstance(config.config_filepath, Path)
    assert hasattr(config, "script_filepath") and isinstance(config.config_filepath, Path)

    # dump python files to reproduce the run
    mlflow.log_artifact(config.config_filepath.as_posix())
    mlflow.log_artifact(config.script_filepath.as_posix())

    output_path = mlflow.get_artifact_uri()
    config.output_path = Path(output_path)

    try:
        training(config, local_rank=local_rank)
    except KeyboardInterrupt:
        logger.info("Catched KeyboardInterrupt -> exit")
    except Exception as e:  # noqa
        logger.exception("")
        mlflow.log_param("Run Status", "FAILED")
        dist.destroy_process_group()
        return

    mlflow.log_param("Run Status", "OK")
    dist.destroy_process_group()
