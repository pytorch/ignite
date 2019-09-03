# This a training script launched with py_config_runner
# It should only contain `run(config, **kwargs)` method

from pathlib import Path

import torch
import torch.distributed as dist

import mlflow

import ignite
from ignite.engine import Events, _prepare_batch
from ignite.metrics import Accuracy, TopKCategoricalAccuracy

from py_config_runner.config_utils import get_params, TRAINVAL_CONFIG, assert_config
from py_config_runner.utils import set_seed


from utils import initialize_amp, setup_distrib_trainer, setup_distrib_evaluators, setup_tb_mlflow_logging, \
    save_best_model_by_val_score, add_early_stopping_by_val_score, setup_distrib_loader


def training(config, local_rank=None):

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

    config.prepare_batch = getattr(config, "prepare_batch", _prepare_batch)
    config.non_blocking = getattr(config, "non_blocking", True)

    model = config.model.to(device)
    optimizer = config.optimizer
    model, optimizer = initialize_amp(model, optimizer, getattr(config, "fp16_opt_level", "O2"))
    criterion = config.criterion.to(device)

    trainer = setup_distrib_trainer(model, optimizer, criterion, train_sampler, device, config)

    def output_transform(output):
        return output['y_pred'], output['y']

    val_metrics = {
        "Accuracy": Accuracy(output_transform=output_transform, device=device),
        "Top-5 Accuracy": TopKCategoricalAccuracy(k=5, output_transform=output_transform, device=device),
    }

    if hasattr(config, "val_metrics") and isinstance(config.val_metrics, dict):
        val_metrics.update(config.val_metrics)

    train_evaluator, evaluator = setup_distrib_evaluators(model, device, val_metrics, config)

    val_interval = getattr(config, "val_interval", 1)

    def run_validation(engine, val_interval):
        if engine.state.epoch > 1 and ((engine.state.epoch - 1) % val_interval == 0):
            train_evaluator.run(train_eval_loader)
            evaluator.run(val_loader)

    trainer.add_event_handler(Events.EPOCH_STARTED, run_validation, val_interval=val_interval)
    trainer.add_event_handler(Events.COMPLETED, run_validation, val_interval=1)

    if dist.get_rank() == 0:

        setup_tb_mlflow_logging(trainer, optimizer, train_evaluator, evaluator, config)

        save_best_model_by_val_score(evaluator, model, metric_name="Accuracy", config=config)

        if hasattr(config, "es_patience"):
            add_early_stopping_by_val_score(evaluator, trainer, metric_name="Accuracy", config=config)

    trainer.run(train_loader, max_epochs=config.num_epochs)


def run(config, logger=None, local_rank=0, **kwargs):

    assert torch.cuda.is_available()
    assert torch.backends.cudnn.enabled, "Nvidia/Amp requires cudnn backend to be enabled."

    dist.init_process_group(getattr(config, "dist_backend", "nccl"),
                            init_method=getattr(config, "dist_url", "env://"))

    # We pass config with option --manual_config_load
    assert hasattr(config, "setup"), "We need setup configuration manually, please set --manual_config_load " \
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
