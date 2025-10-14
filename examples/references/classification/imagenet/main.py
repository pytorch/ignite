import os
from functools import partial
from pathlib import Path

import fire
import torch

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    raise RuntimeError("Please, use recent PyTorch version, e.g. >=1.12.0")

import dataflow as data
import utils
import vis
from py_config_runner import ConfigObject, get_params, InferenceConfigSchema, TrainvalConfigSchema

import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, Timer
from ignite.metrics import Accuracy, Frequency, TopKCategoricalAccuracy
from ignite.utils import manual_seed, setup_logger


def training(local_rank, config, logger, with_clearml):
    rank = idist.get_rank()
    manual_seed(config.seed + local_rank)

    train_loader = config.train_loader
    val_loader = config.val_loader
    train_eval_loader = config.train_eval_loader

    model, optimizer, criterion = utils.initialize(config)

    # Setup trainer for this specific task
    trainer = create_trainer(model, optimizer, criterion, train_loader.sampler, config, logger, with_clearml)

    # Setup evaluators
    accuracy = Accuracy()
    val_metrics = {
        "Accuracy": accuracy,
        "Top-5 Accuracy": TopKCategoricalAccuracy(k=5),
        "Error": (1.0 - accuracy) * 100,
    }

    if ("val_metrics" in config) and isinstance(config.val_metrics, dict):
        val_metrics.update(config.val_metrics)

    evaluator = create_evaluator(model, val_metrics, config, with_clearml, tag="val")
    train_evaluator = create_evaluator(model, val_metrics, config, with_clearml, tag="train")

    val_interval = config.get("val_interval", 1)

    # Run validation on every val_interval epoch, in the end of the training
    # and in the begining if config.start_by_validation is True
    event = Events.EPOCH_COMPLETED(every=val_interval)
    if config.num_epochs % val_interval != 0:
        event |= Events.COMPLETED
    if config.get("start_by_validation", False):
        event |= Events.STARTED

    @trainer.on(event)
    def run_validation():
        epoch = trainer.state.epoch
        state = train_evaluator.run(train_eval_loader)
        utils.log_metrics(logger, epoch, state.times["COMPLETED"], "Train", state.metrics)
        state = evaluator.run(val_loader)
        utils.log_metrics(logger, epoch, state.times["COMPLETED"], "Test", state.metrics)

    score_metric_name = "Accuracy"
    if "es_patience" in config:
        common.add_early_stopping_by_val_score(config.es_patience, evaluator, trainer, metric_name=score_metric_name)

    # Store 2 best models by validation accuracy:
    common.gen_save_best_models_by_val_score(
        save_handler=utils.get_save_handler(config.output_path.as_posix(), with_clearml),
        evaluator=evaluator,
        models=model,
        metric_name=score_metric_name,
        n_saved=2,
        trainer=trainer,
        tag="val",
    )

    # Setup Tensorboard logger
    if rank == 0:
        tb_logger = common.setup_tb_logging(
            config.output_path.as_posix(),
            trainer,
            optimizer,
            evaluators={"training": train_evaluator, "validation": evaluator},
        )

        # Log validation predictions as images
        # We define a custom event filter to log less frequently the images (to reduce storage size)
        # - we plot images with masks of the middle validation batch
        # - once every 3 validations and
        # - at the end of the training
        def custom_event_filter(_, val_iteration):
            c1 = val_iteration == 1
            c2 = trainer.state.epoch % (config.get("val_interval", 1) * 3) == 0
            c2 |= trainer.state.epoch == config.num_epochs
            return c1 and c2

        # Image denormalization function to plot predictions with images
        mean = config.get("mean", (0.485, 0.456, 0.406))
        std = config.get("std", (0.229, 0.224, 0.225))
        img_denormalize = partial(data.denormalize, mean=mean, std=std)

        tb_logger.attach(
            evaluator,
            log_handler=vis.predictions_gt_images_handler(
                img_denormalize_fn=img_denormalize, n_images=12, another_engine=trainer, prefix_tag="validation"
            ),
            event_name=Events.ITERATION_COMPLETED(event_filter=custom_event_filter),
        )

        tb_logger.attach(
            train_evaluator,
            log_handler=vis.predictions_gt_images_handler(
                img_denormalize_fn=img_denormalize, n_images=12, another_engine=trainer, prefix_tag="training"
            ),
            event_name=Events.ITERATION_COMPLETED(event_filter=custom_event_filter),
        )

    trainer.run(train_loader, max_epochs=config.num_epochs)

    if idist.get_rank() == 0:
        tb_logger.close()


def create_trainer(model, optimizer, criterion, train_sampler, config, logger, with_clearml):
    device = config.device
    prepare_batch = data.prepare_batch

    # Setup trainer
    accumulation_steps = config.get("accumulation_steps", 1)
    model_output_transform = config.get("model_output_transform", lambda x: x)

    with_amp = config.get("with_amp", True)
    scaler = GradScaler(enabled=with_amp)

    def training_step(engine, batch):
        model.train()
        x, y = prepare_batch(batch, device=device, non_blocking=True)
        with autocast("cuda", enabled=with_amp):
            y_pred = model(x)
            y_pred = model_output_transform(y_pred)
            loss = criterion(y_pred, y) / accumulation_steps

        output = {"supervised batch loss": loss.item(), "num_samples": len(x)}

        scaler.scale(loss).backward()
        if engine.state.iteration % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        return output

    trainer = Engine(training_step)
    trainer.logger = logger

    throughput_metric = Frequency(output_transform=lambda x: x["num_samples"])
    throughput_metric.attach(trainer, name="Throughput")

    timer = Timer(average=True)
    timer.attach(
        trainer,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=20))
    def log_progress():
        metrics = dict(trainer.state.metrics)
        epoch_length = trainer.state.epoch_length

        metrics["ETA (seconds)"] = int((epoch_length - (trainer.state.iteration % epoch_length)) * timer.value())

        metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        metrics_format = (
            f"[{trainer.state.epoch}/{trainer.state.max_epochs}] "
            + f"Iter={trainer.state.iteration % epoch_length}/{epoch_length}: "
            + f"{metrics_str}"
        )
        trainer.logger.info(metrics_format)

    output_names = [
        "supervised batch loss",
    ]
    lr_scheduler = config.lr_scheduler

    to_save = {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "trainer": trainer,
        "amp": scaler,
    }

    save_every_iters = config.get("save_every_iters", 1000)

    common.setup_common_training_handlers(
        trainer,
        train_sampler,
        to_save=to_save,
        save_every_iters=save_every_iters,
        save_handler=utils.get_save_handler(config.output_path.as_posix(), with_clearml),
        lr_scheduler=lr_scheduler,
        output_names=output_names,
        # with_pbars=not with_clearml,
        with_pbars=False,
        log_every_iters=1,
    )

    resume_from = config.get("resume_from", None)
    if resume_from is not None:
        checkpoint_fp = Path(resume_from)
        assert checkpoint_fp.exists(), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
        logger.info(f"Resume from a checkpoint: {checkpoint_fp.as_posix()}")
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    return trainer


def create_evaluator(model, metrics, config, with_clearml, tag="val"):
    model_output_transform = config.get("model_output_transform", lambda x: x)
    with_amp = config.get("with_amp", True)
    prepare_batch = data.prepare_batch

    @torch.no_grad()
    def evaluate_step(engine, batch):
        model.eval()
        with autocast("cuda", enabled=with_amp):
            x, y = prepare_batch(batch, device=config.device, non_blocking=True)
            y_pred = model(x)
            y_pred = model_output_transform(y_pred)
        return y_pred, y

    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    if idist.get_rank() == 0 and (not with_clearml):
        common.ProgressBar(desc=f"Evaluation ({tag})", persist=False).attach(evaluator)

    return evaluator


def setup_experiment_tracking(config, with_clearml, task_type="training"):
    from datetime import datetime

    assert task_type in ("training", "testing"), task_type

    output_path = ""
    if idist.get_rank() == 0:
        if with_clearml:
            from clearml import Task

            schema = TrainvalConfigSchema if task_type == "training" else InferenceConfigSchema

            task = Task.init("ImageNet Training", config.config_filepath.stem, task_type=task_type)
            task.connect_configuration(config.config_filepath.as_posix())

            task.upload_artifact(config.script_filepath.name, config.script_filepath.as_posix())
            task.upload_artifact(config.config_filepath.name, config.config_filepath.as_posix())
            task.connect(get_params(config, schema))

            output_path = Path(os.environ.get("CLEARML_OUTPUT_PATH", "/tmp"))
            output_path = output_path / "clearml" / datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            import shutil

            output_path = Path(os.environ.get("OUTPUT_PATH", "/tmp/output-imagenet"))
            output_path = output_path / task_type / config.config_filepath.stem
            output_path = output_path / datetime.now().strftime("%Y%m%d-%H%M%S")
            output_path.mkdir(parents=True, exist_ok=True)

            shutil.copyfile(config.script_filepath.as_posix(), output_path / config.script_filepath.name)
            shutil.copyfile(config.config_filepath.as_posix(), output_path / config.config_filepath.name)

        output_path = output_path.as_posix()
    return Path(idist.broadcast(output_path, src=0))


def run_training(config_filepath, backend="nccl", with_clearml=True):
    """Main entry to run training experiment

    Args:
        config_filepath (str): training configuration .py file
        backend (str): distributed backend: nccl, gloo or None to run without distributed config
        with_clearml (bool): if True, uses ClearML as experiment tracking system
    """
    assert torch.cuda.is_available(), torch.cuda.is_available()
    assert torch.backends.cudnn.enabled
    torch.backends.cudnn.benchmark = True

    config_filepath = Path(config_filepath)
    assert config_filepath.exists(), f"File '{config_filepath.as_posix()}' is not found"

    with idist.Parallel(backend=backend) as parallel:
        logger = setup_logger(name="ImageNet Training", distributed_rank=idist.get_rank())

        config = ConfigObject(config_filepath)
        TrainvalConfigSchema.validate(config)
        config.script_filepath = Path(__file__)

        output_path = setup_experiment_tracking(config, with_clearml=with_clearml)
        config.output_path = output_path

        utils.log_basic_info(logger, get_params(config, TrainvalConfigSchema))

        try:
            parallel.run(training, config, logger=logger, with_clearml=with_clearml)
        except KeyboardInterrupt:
            logger.info("Catched KeyboardInterrupt -> exit")
        except Exception as e:  # noqa
            logger.exception("")
            raise e


def get_model_weights(config, logger, with_clearml):
    path = ""
    if with_clearml:
        from clearml import Model

        if idist.get_rank() > 0:
            idist.barrier()
        else:
            model_id = config.weights_path

            logger.info(f"Loading trained model: {model_id}")
            model = Model(model_id)
            assert model is not None, f"{model_id}"
            path = model.get_local_copy()
            idist.barrier()
        path = idist.broadcast(path, src=0)
    else:
        path = config.weights_path
        logger.info(f"Loading {path}")

    assert Path(path).exists(), f"{path} is not found"
    return torch.load(path)


def evaluation(local_rank, config, logger, with_clearml):
    rank = idist.get_rank()
    device = idist.device()
    manual_seed(config.seed + local_rank)

    data_loader = config.data_loader
    model = config.model.to(device)

    # Load weights:
    state_dict = get_model_weights(config, logger, with_clearml)
    model.load_state_dict(state_dict)

    # Adapt model to dist config
    model = idist.auto_model(model)

    # Setup evaluators
    val_metrics = {
        "Accuracy": Accuracy(),
        "Top-5 Accuracy": TopKCategoricalAccuracy(k=5),
    }

    if ("val_metrics" in config) and isinstance(config.val_metrics, dict):
        val_metrics.update(config.val_metrics)

    evaluator = create_evaluator(model, val_metrics, config, with_clearml, tag="val")

    # Setup Tensorboard logger
    if rank == 0:
        tb_logger = common.TensorboardLogger(log_dir=config.output_path.as_posix())
        tb_logger.attach_output_handler(evaluator, event_name=Events.COMPLETED, tag="validation", metric_names="all")

    state = evaluator.run(data_loader)
    utils.log_metrics(logger, 0, state.times["COMPLETED"], "Validation", state.metrics)

    if idist.get_rank() == 0:
        tb_logger.close()


def run_evaluation(config_filepath, backend="nccl", with_clearml=True):
    """Main entry to run model's evaluation:
        - compute validation metrics

    Args:
        config_filepath (str): evaluation configuration .py file
        backend (str): distributed backend: nccl, gloo, horovod or None to run without distributed config
        with_clearml (bool): if True, uses ClearML as experiment tracking system
    """
    assert torch.cuda.is_available(), torch.cuda.is_available()
    assert torch.backends.cudnn.enabled
    torch.backends.cudnn.benchmark = True

    config_filepath = Path(config_filepath)
    assert config_filepath.exists(), f"File '{config_filepath.as_posix()}' is not found"

    with idist.Parallel(backend=backend) as parallel:
        logger = setup_logger(name="ImageNet Evaluation", distributed_rank=idist.get_rank())

        config = ConfigObject(config_filepath)
        InferenceConfigSchema.validate(config)
        config.script_filepath = Path(__file__)

        output_path = setup_experiment_tracking(config, with_clearml=with_clearml, task_type="testing")
        config.output_path = output_path

        utils.log_basic_info(logger, get_params(config, InferenceConfigSchema))

        try:
            parallel.run(evaluation, config, logger=logger, with_clearml=with_clearml)
        except KeyboardInterrupt:
            logger.info("Catched KeyboardInterrupt -> exit")
        except Exception as e:  # noqa
            logger.exception("")
            raise e


if __name__ == "__main__":
    fire.Fire({"training": run_training, "eval": run_evaluation})
