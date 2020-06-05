import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist


import ignite
from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.engine.deterministic import DeterministicEngine
from ignite.metrics import Accuracy, Loss
from ignite.handlers import Checkpoint, global_step_from_engine
from ignite.utils import convert_tensor, manual_seed

from ignite.contrib.engines import common
from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler, GradsHistHandler
from ignite.contrib.handlers import PiecewiseLinear

import ignite.distributed as idist

import utils


@idist.auto_distributed(backend="gloo")
def run(config):

    # distributed = dist.is_available() and dist.is_initialized()
    # rank = dist.get_rank() if distributed else 0

    distributed = idist.world_size() > 0
    rank = idist.rank()

    if distributed:
        # let each node print the info
        if idist.local_rank() == 0:
            print("\nDistributed setting:")
            print("\tbackend: {}".format(dist.get_backend()))
            print("\tworld size: {}".format(dist.get_world_size()))
            print("\trank: {}".format(dist.get_rank()))
            print("\n")

    output_path = None
    # let each node print the info
    if idist.local_rank() == 0:
        print("Train {} on CIFAR10".format(config["model"]))
        print("- PyTorch version: {}".format(torch.__version__))
        print("- Ignite version: {}".format(ignite.__version__))
        print("- CUDA version: {}".format(torch.version.cuda))

        print("\n")
        print("Configuration:")
        for key, value in config.items():
            print("\t{}: {}".format(key, value))
        print("\n")

        # create log directory only by 1 node
        if (not distributed) or (dist.get_rank() == 0):
            from datetime import datetime

            now = datetime.now().strftime("%Y%m%d-%H%M%S")
            gpu_conf = "-single-gpu"
            if distributed:
                ngpus_per_node = torch.cuda.device_count()
                nnodes = dist.get_world_size() // ngpus_per_node
                gpu_conf = "-distributed-{}nodes-{}gpus".format(nnodes, ngpus_per_node)

            output_path = Path(config["output_path"]) / "{}{}".format(now, gpu_conf)
            if not output_path.exists():
                output_path.mkdir(parents=True)
            output_path = output_path.as_posix()
            print("Output path: {}".format(output_path))

    # show the distributed config
    idist.show_config()

    manual_seed(config["seed"] + rank)

    # Setup dataflow, model, optimizer, criterion
    train_loader, test_loader = utils.get_dataflow(config, distributed)
    model, optimizer = utils.get_model_optimizer(config, distributed)

    # TODO check for tpu...
    device = idist.device()

    criterion = nn.CrossEntropyLoss().to(device)

    le = len(train_loader)
    milestones_values = [
        (0, 0.0),
        (le * config["num_warmup_epochs"], config["learning_rate"]),
        (le * config["num_epochs"], 0.0),
    ]
    lr_scheduler = PiecewiseLinear(optimizer, param_name="lr", milestones_values=milestones_values)

    # Setup Ignite trainer:
    # - let's define training step
    # - add other common handlers:
    #    - TerminateOnNan,
    #    - handler to setup learning rate scheduling,
    #    - ModelCheckpoint
    #    - RunningAverage` on `train_step` output
    #    - Two progress bars on epochs and optionally on iterations

    def train_step(engine, batch):

        x = convert_tensor(batch[0], device=device, non_blocking=True)
        y = convert_tensor(batch[1], device=device, non_blocking=True)

        model.train()
        # Supervised part
        y_pred = model(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            "batch loss": loss.item(),
        }

    if config["deterministic"] and rank == 0:
        print("Setup deterministic trainer")
    trainer = Engine(train_step) if not config["deterministic"] else DeterministicEngine(train_step)
    train_sampler = train_loader.sampler if distributed else None
    to_save = {"trainer": trainer, "model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler}
    metric_names = [
        "batch loss",
    ]
    common.setup_common_training_handlers(
        trainer,
        train_sampler=train_sampler,
        to_save=to_save,
        save_every_iters=config["checkpoint_every"],
        output_path=output_path,
        lr_scheduler=lr_scheduler,
        output_names=metric_names,
        with_pbar_on_iters=config["display_iters"],
        log_every_iters=10,
    )

    if rank == 0:
        # Setup Tensorboard logger - wrapper on SummaryWriter
        tb_logger = TensorboardLogger(log_dir=output_path)
        # Attach logger to the trainer and log trainer's metrics (stored in trainer.state.metrics) every iteration
        tb_logger.attach(
            trainer,
            log_handler=OutputHandler(tag="train", metric_names=metric_names),
            event_name=Events.ITERATION_COMPLETED,
        )
        # log optimizer's parameters: "lr" every iteration
        tb_logger.attach(
            trainer, log_handler=OptimizerParamsHandler(optimizer, param_name="lr"), event_name=Events.ITERATION_STARTED
        )

    # Let's now setup evaluator engine to perform model's validation and compute metrics
    metrics = {
        "accuracy": Accuracy(device=device),
        "loss": Loss(criterion, device=device),
    }

    # We define two evaluators as they wont have exactly similar roles:
    # - `evaluator` will save the best model based on validation score
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)

    def run_validation(engine):
        train_evaluator.run(train_loader)
        evaluator.run(test_loader)

    trainer.add_event_handler(Events.EPOCH_STARTED(every=config["validate_every"]), run_validation)
    trainer.add_event_handler(Events.COMPLETED, run_validation)

    if rank == 0:
        # Setup progress bar on evaluation engines
        if config["display_iters"]:
            ProgressBar(persist=False, desc="Train evaluation").attach(train_evaluator)
            ProgressBar(persist=False, desc="Test evaluation").attach(evaluator)

        # Let's log metrics of `train_evaluator` stored in `train_evaluator.state.metrics` when validation run is done
        tb_logger.attach(
            train_evaluator,
            log_handler=OutputHandler(
                tag="train", metric_names="all", global_step_transform=global_step_from_engine(trainer)
            ),
            event_name=Events.COMPLETED,
        )

        # Let's log metrics of `evaluator` stored in `evaluator.state.metrics` when validation run is done
        tb_logger.attach(
            evaluator,
            log_handler=OutputHandler(
                tag="test", metric_names="all", global_step_transform=global_step_from_engine(trainer)
            ),
            event_name=Events.COMPLETED,
        )

        # Store 3 best models by validation accuracy:
        common.save_best_model_by_val_score(
            output_path, evaluator, model=model, metric_name="accuracy", n_saved=3, trainer=trainer, tag="test"
        )

        # Optionally log model gradients
        if config["log_model_grads_every"] is not None:
            tb_logger.attach(
                trainer,
                log_handler=GradsHistHandler(model, tag=model.__class__.__name__),
                event_name=Events.ITERATION_COMPLETED(every=config["log_model_grads_every"]),
            )

    # In order to check training resuming we can emulate a crash
    if config["crash_iteration"] is not None:

        @trainer.on(Events.ITERATION_STARTED(once=config["crash_iteration"]))
        def _(engine):
            raise Exception("STOP at iteration: {}".format(engine.state.iteration))

    resume_from = config["resume_from"]
    if resume_from is not None:
        checkpoint_fp = Path(resume_from)
        assert checkpoint_fp.exists(), "Checkpoint '{}' is not found".format(checkpoint_fp.as_posix())
        print("Resume from a checkpoint: {}".format(checkpoint_fp.as_posix()))
        checkpoint = torch.load(checkpoint_fp.as_posix())
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    try:
        trainer.run(train_loader, max_epochs=config["num_epochs"])
    except Exception as e:
        import traceback

        print(traceback.format_exc())

    if rank == 0:
        tb_logger.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Training a CNN on CIFAR10 dataset")
    parser.add_argument(
        "--params",
        type=str,
        help="Override default configuration with parameters: "
        "data_path=/path/to/dataset;batch_size=64;num_workers=12 ...",
    )
    # parser.add_argument("--local_rank", type=int, default=0, help="Local process rank in distributed computation")

    args = parser.parse_args()

    # assert torch.cuda.is_available()
    # torch.backends.cudnn.benchmark = True

    config = utils.get_default_config()
    # config["local_rank"] = args.local_rank

    # Override config:
    if args.params is not None:
        for param in args.params.split(";"):
            key, value = param.split("=")
            if "/" not in value:
                value = eval(value)
            config[key] = value

    # backend = config["dist_backend"]
    # distributed = backend is not None

    # if distributed:
    #     dist.init_process_group(backend, init_method=config["dist_url"])
    #     # let each node print the info
    #     if config["local_rank"] == 0:
    #         print("\nDistributed setting:")
    #         print("\tbackend: {}".format(dist.get_backend()))
    #         print("\tworld size: {}".format(dist.get_world_size()))
    #         print("\trank: {}".format(dist.get_rank()))
    #         print("\n")

    # output_path = None
    # # let each node print the info
    # if config["local_rank"] == 0:
    #     print("Train {} on CIFAR10".format(config["model"]))
    #     print("- PyTorch version: {}".format(torch.__version__))
    #     print("- Ignite version: {}".format(ignite.__version__))
    #     print("- CUDA version: {}".format(torch.version.cuda))
    #
    #     print("\n")
    #     print("Configuration:")
    #     for key, value in config.items():
    #         print("\t{}: {}".format(key, value))
    #     print("\n")
    #
    #     # create log directory only by 1 node
    #     if (not distributed) or (dist.get_rank() == 0):
    #         from datetime import datetime
    #
    #         now = datetime.now().strftime("%Y%m%d-%H%M%S")
    #         gpu_conf = "-single-gpu"
    #         if distributed:
    #             ngpus_per_node = torch.cuda.device_count()
    #             nnodes = dist.get_world_size() // ngpus_per_node
    #             gpu_conf = "-distributed-{}nodes-{}gpus".format(nnodes, ngpus_per_node)
    #
    #         output_path = Path(config["output_path"]) / "{}{}".format(now, gpu_conf)
    #         if not output_path.exists():
    #             output_path.mkdir(parents=True)
    #         output_path = output_path.as_posix()
    #         print("Output path: {}".format(output_path))
    #
    # try:
    #     run(output_path, config)
    # except KeyboardInterrupt:
    #     print("Catched KeyboardInterrupt -> exit")
    # except Exception as e:
    #     if distributed:
    #         dist.destroy_process_group()
    #     raise e
    #
    # if distributed:
    #     dist.destroy_process_group()

    run(config)
