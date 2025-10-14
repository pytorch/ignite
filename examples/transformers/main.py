import os
from datetime import datetime
from pathlib import Path

import fire
import torch
import torch.nn as nn
import torch.optim as optim
import utils
from torch.amp import autocast, GradScaler

import ignite
import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, global_step_from_engine, PiecewiseLinear
from ignite.metrics import Accuracy, Loss
from ignite.utils import manual_seed, setup_logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # remove tokenizer paralleism warning


def training(local_rank, config):
    rank = idist.get_rank()
    manual_seed(config["seed"] + rank)
    device = idist.device()

    logger = setup_logger(name="IMDB-Training", distributed_rank=local_rank)

    log_basic_info(logger, config)

    output_path = config["output_dir"]
    if rank == 0:
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"{config['model']}_backend-{idist.backend()}-{idist.get_world_size()}_{now}"
        output_path = Path(output_path) / folder_name
        if not output_path.exists():
            output_path.mkdir(parents=True)
        config["output_dir"] = output_path.as_posix()
        logger.info(f"Output path: {config['output_dir']}")

        if "cuda" in device.type:
            config["cuda device name"] = torch.cuda.get_device_name(local_rank)

        if config["with_clearml"]:
            from clearml import Task

            task = Task.init("IMDB-Training", task_name=output_path.stem)
            task.connect_configuration(config)
            # Log hyper parameters
            hyper_params = [
                "model",
                "dropout",
                "n_fc",
                "batch_size",
                "max_length",
                "weight_decay",
                "num_epochs",
                "learning_rate",
                "num_warmup_epochs",
            ]
            task.connect({k: config[k] for k in hyper_params})

    # Setup dataflow, model, optimizer, criterion
    train_loader, test_loader = get_dataflow(config)

    config["num_iters_per_epoch"] = len(train_loader)
    model, optimizer, criterion, lr_scheduler = initialize(config)

    # Create trainer for current task
    trainer = create_trainer(model, optimizer, criterion, lr_scheduler, train_loader.sampler, config, logger)

    # Let's now setup evaluator engine to perform model's validation and compute metrics
    metrics = {
        "Accuracy": Accuracy(output_transform=utils.thresholded_output_transform),
        "Loss": Loss(criterion),
    }

    # We define two evaluators as they wont have exactly similar roles:
    # - `evaluator` will save the best model based on validation score
    evaluator = create_evaluator(model, metrics, config, tag="val")
    train_evaluator = create_evaluator(model, metrics, config, tag="train")

    def run_validation(engine):
        epoch = trainer.state.epoch
        state = train_evaluator.run(train_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Train", state.metrics)
        state = evaluator.run(test_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "Test", state.metrics)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config["validate_every"]) | Events.COMPLETED | Events.STARTED, run_validation
    )

    if rank == 0:
        # Setup TensorBoard logging on trainer and evaluators. Logged values are:
        #  - Training metrics, e.g. running average loss values
        #  - Learning rate
        #  - Evaluation train/test metrics
        evaluators = {"training": train_evaluator, "test": evaluator}
        tb_logger = common.setup_tb_logging(
            output_path, trainer, optimizer, evaluators=evaluators, log_every_iters=config["log_every_iters"]
        )

    # Store 2 best models by validation accuracy starting from num_epochs / 2:
    best_model_handler = Checkpoint(
        {"model": model},
        utils.get_save_handler(config),
        filename_prefix="best",
        n_saved=2,
        global_step_transform=global_step_from_engine(trainer),
        score_name="test_accuracy",
        score_function=Checkpoint.get_default_score_fn("Accuracy"),
    )
    evaluator.add_event_handler(
        Events.COMPLETED(lambda *_: trainer.state.epoch > config["num_epochs"] // 2), best_model_handler
    )

    try:
        trainer.run(train_loader, max_epochs=config["num_epochs"])
    except Exception as e:
        logger.exception("")
        raise e

    if rank == 0:
        tb_logger.close()


def run(
    seed=543,
    data_dir="/tmp/data",
    output_dir="/tmp/output-imdb/",
    model="bert-base-uncased",
    model_dir="/tmp/model",
    tokenizer_dir="/tmp/tokenizer",
    num_classes=1,
    dropout=0.3,
    n_fc=768,
    max_length=256,
    batch_size=32,
    weight_decay=0.01,
    num_workers=4,
    num_epochs=3,
    learning_rate=5e-5,
    num_warmup_epochs=0,
    validate_every=1,
    checkpoint_every=1000,
    backend=None,
    resume_from=None,
    log_every_iters=15,
    nproc_per_node=None,
    with_clearml=False,
    with_amp=False,
    **spawn_kwargs,
):
    """Main entry to fintune a transformer model on the IMDB dataset for sentiment classification.
    Args:
        seed (int): random state seed to set. Default, 543.
        data_dir (str): dataset cache directory. Default, "/tmp/data".
        output_path (str): output path. Default, "/tmp/output-IMDB".
        model (str): model name (from transformers) to setup model,tokenize and config to train. Default,
        "bert-base-uncased".
        model_dir (str): cache directory to download the pretrained model. Default, "/tmp/model".
        tokenizer_dir (str) : tokenizer cache directory. Default, "/tmp/tokenizer".
        num_classes (int) : number of target classes. Default, 1 (binary classification).
        dropout (float) : dropout probability. Default, 0.3.
        n_fc (int) : number of neurons in the last fully connected layer. Default, 768.
        max_length (int) : maximum number of tokens for the inputs to the transformer model. Default,256
        batch_size (int): total batch size. Default, 128 .
        weight_decay (float): weight decay. Default, 0.01 .
        num_workers (int): number of workers in the data loader. Default, 12.
        num_epochs (int): number of epochs to train the model. Default, 5.
        learning_rate (float): peak of piecewise linear learning rate scheduler. Default, 5e-5.
        num_warmup_epochs (int): number of warm-up epochs before learning rate decay. Default, 3.
        validate_every (int): run model's validation every ``validate_every`` epochs. Default, 3.
        checkpoint_every (int): store training checkpoint every ``checkpoint_every`` iterations. Default, 1000.
        backend (str, optional): backend to use for distributed configuration. Possible values: None, "nccl", "xla-tpu",
            "gloo" etc. Default, None.
        nproc_per_node (int, optional): optional argument to setup number of processes per node. It is useful,
            when main python process is spawning training as child processes.
        resume_from (str, optional): path to checkpoint to use to resume the training from. Default, None.
        log_every_iters (int): argument to log batch loss every ``log_every_iters`` iterations.
            It can be 0 to disable it. Default, 15.
        with_clearml (bool): if True, experiment ClearML logger is setup. Default, False.
        with_amp (bool): if True, enables native automatic mixed precision. Default, False.
        **spawn_kwargs: Other kwargs to spawn run in child processes: master_addr, master_port, node_rank, nnodes
    """
    # check to see if the num_epochs is greater than or equal to num_warmup_epochs
    if num_warmup_epochs >= num_epochs:
        raise ValueError(
            "num_epochs cannot be less than or equal to num_warmup_epochs, please increase num_epochs or decrease "
            "num_warmup_epochs"
        )

    # catch all local parameters
    config = locals()
    config.update(config["spawn_kwargs"])
    del config["spawn_kwargs"]

    spawn_kwargs["nproc_per_node"] = nproc_per_node

    with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:
        parallel.run(training, config)


def get_dataflow(config):
    # - Get train/test datasets
    if idist.get_local_rank() > 0:
        # Ensure that only local rank 0 download the dataset
        # Thus each node will download a copy of the dataset
        idist.barrier()

    train_dataset, test_dataset = utils.get_dataset(
        config["data_dir"], config["model"], config["tokenizer_dir"], config["max_length"]
    )

    if idist.get_local_rank() == 0:
        # Ensure that only local rank 0 download the dataset
        idist.barrier()

    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    train_loader = idist.auto_dataloader(
        train_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"], shuffle=True, drop_last=True
    )

    test_loader = idist.auto_dataloader(
        test_dataset, batch_size=2 * config["batch_size"], num_workers=config["num_workers"], shuffle=False
    )
    return train_loader, test_loader


def initialize(config):
    model = utils.get_model(
        config["model"], config["model_dir"], config["dropout"], config["n_fc"], config["num_classes"]
    )

    config["learning_rate"] *= idist.get_world_size()
    # Adapt model for distributed settings if configured
    model = idist.auto_model(model)

    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    optimizer = idist.auto_optim(optimizer)
    criterion = nn.BCEWithLogitsLoss()

    le = config["num_iters_per_epoch"]
    milestones_values = [
        (0, 0.0),
        (le * config["num_warmup_epochs"], config["learning_rate"]),
        (le * config["num_epochs"], 0.0),
    ]
    lr_scheduler = PiecewiseLinear(optimizer, param_name="lr", milestones_values=milestones_values)

    return model, optimizer, criterion, lr_scheduler


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}")


def log_basic_info(logger, config):
    logger.info(f"Train {config['model']} on IMDB")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as
        # torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")


def create_trainer(model, optimizer, criterion, lr_scheduler, train_sampler, config, logger):
    device = idist.device()

    # Setup Ignite trainer:
    # - let's define training step
    # - add other common handlers:
    #    - TerminateOnNan,
    #    - handler to setup learning rate scheduling,
    #    - ModelCheckpoint
    #    - RunningAverage` on `train_step` output
    #    - Two progress bars on epochs and optionally on iterations

    with_amp = config["with_amp"]
    scaler = GradScaler(enabled=with_amp)

    def train_step(engine, batch):
        input_batch = batch[0]
        labels = batch[1].view(-1, 1)

        if labels.device != device:
            input_batch = {k: v.to(device, non_blocking=True, dtype=torch.long) for k, v in batch[0].items()}
            labels = labels.to(device, non_blocking=True, dtype=torch.float)

        model.train()

        with autocast("cuda", enabled=with_amp):
            y_pred = model(input_batch)
            loss = criterion(y_pred, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return {
            "batch loss": loss.item(),
        }

    trainer = Engine(train_step)
    trainer.logger = logger

    to_save = {"trainer": trainer, "model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler}
    metric_names = [
        "batch loss",
    ]
    if config["log_every_iters"] == 0:
        # Disable logging training metrics:
        metric_names = None
        config["log_every_iters"] = 15

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        to_save=to_save,
        save_every_iters=config["checkpoint_every"],
        save_handler=utils.get_save_handler(config),
        lr_scheduler=lr_scheduler,
        output_names=metric_names,
        log_every_iters=config["log_every_iters"],
        with_pbars=not config["with_clearml"],
        clear_cuda_cache=False,
    )

    resume_from = config["resume_from"]
    if resume_from is not None:
        checkpoint_fp = Path(resume_from)
        assert checkpoint_fp.exists(), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
        logger.info(f"Resume from a checkpoint: {checkpoint_fp.as_posix()}")
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    return trainer


def create_evaluator(model, metrics, config, tag="val"):
    with_amp = config["with_amp"]
    device = idist.device()

    @torch.no_grad()
    def evaluate_step(engine, batch):
        model.eval()

        input_batch = batch[0]
        labels = batch[1].view(-1, 1)

        if labels.device != device:
            input_batch = {k: v.to(device, non_blocking=True, dtype=torch.long) for k, v in batch[0].items()}
            labels = labels.to(device, non_blocking=True, dtype=torch.float)

        with autocast("cuda", enabled=with_amp):
            output = model(input_batch)
        return output, labels

    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    if idist.get_rank() == 0 and (not config["with_clearml"]):
        common.ProgressBar(desc=f"Evaluation ({tag})", persist=False).attach(evaluator)

    return evaluator


if __name__ == "__main__":
    fire.Fire({"run": run})
