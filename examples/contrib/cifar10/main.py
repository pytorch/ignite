import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.parallel
import torch.distributed as dist

import ignite
from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import Checkpoint, global_step_from_engine
from ignite.utils import convert_tensor

from ignite.contrib.engines import common
from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler, GradsHistHandler

from ignite.contrib.handlers import PiecewiseLinear

from utils import set_seed, get_train_test_loaders, get_model


def run(output_path, config):
    device = "cuda"

    local_rank = config['local_rank']
    distributed = backend is not None
    if distributed:
        torch.cuda.set_device(local_rank)
        device = "cuda"
    rank = dist.get_rank() if distributed else 0

    torch.manual_seed(config['seed'] + rank)

    # Rescale batch_size and num_workers
    ngpus_per_node = torch.cuda.device_count()
    ngpus = dist.get_world_size() if distributed else 1
    batch_size = config['batch_size'] // ngpus
    num_workers = int((config['num_workers'] + ngpus_per_node - 1) / ngpus_per_node)

    train_loader, test_loader = get_train_test_loaders(
        path=config['data_path'],
        batch_size=batch_size,
        distributed=distributed,
        num_workers=num_workers
    )

    model = get_model(config['model'])
    model = model.to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank, ],
                                                          output_device=local_rank)

    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'],
                          momentum=config['momentum'],
                          weight_decay=config['weight_decay'],
                          nesterov=True)

    criterion = nn.CrossEntropyLoss().to(device)

    le = len(train_loader)
    milestones_values = [
        (0, 0.0),
        (le * config['num_warmup_epochs'], config['learning_rate']),
        (le * config['num_epochs'], 0.0)
    ]
    lr_scheduler = PiecewiseLinear(optimizer, param_name="lr",
                                   milestones_values=milestones_values)

    def _prepare_batch(batch, device, non_blocking):
        x, y = batch
        return (convert_tensor(x, device=device, non_blocking=non_blocking),
                convert_tensor(y, device=device, non_blocking=non_blocking))

    def process_function(engine, batch):

        x, y = _prepare_batch(batch, device=device, non_blocking=True)

        model.train()
        # Supervised part
        y_pred = model(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            'batch loss': loss.item(),
        }

    trainer = Engine(process_function)
    train_sampler = train_loader.sampler if distributed else None
    to_save = {'trainer': trainer, 'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
    metric_names = ['batch loss', ]
    common.setup_common_training_handlers(trainer, train_sampler=train_sampler,
                                          to_save=to_save, save_every_iters=config['checkpoint_every'],
                                          output_path=output_path, lr_scheduler=lr_scheduler,
                                          output_names=metric_names, with_pbar_on_iters=config['display_iters'],
                                          log_every_iters=10)

    if rank == 0:
        tb_logger = TensorboardLogger(log_dir=output_path)
        tb_logger.attach(trainer,
                         log_handler=OutputHandler(tag="train",
                                                   metric_names=metric_names),
                         event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer,
                         log_handler=OptimizerParamsHandler(optimizer, param_name="lr"),
                         event_name=Events.ITERATION_STARTED)

    metrics = {
        "accuracy": Accuracy(device=device if distributed else None),
        "loss": Loss(criterion, device=device if distributed else None)
    }

    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)

    def run_validation(engine):
        torch.cuda.synchronize()
        train_evaluator.run(train_loader)
        evaluator.run(test_loader)

    trainer.add_event_handler(Events.EPOCH_STARTED(every=config['validate_every']), run_validation)
    trainer.add_event_handler(Events.COMPLETED, run_validation)

    if rank == 0:
        if config['display_iters']:
            ProgressBar(persist=False, desc="Train evaluation").attach(train_evaluator)
            ProgressBar(persist=False, desc="Test evaluation").attach(evaluator)

        tb_logger.attach(train_evaluator,
                         log_handler=OutputHandler(tag="train",
                                                   metric_names=list(metrics.keys()),
                                                   global_step_transform=global_step_from_engine(trainer)),
                         event_name=Events.COMPLETED)

        tb_logger.attach(evaluator,
                         log_handler=OutputHandler(tag="test",
                                                   metric_names=list(metrics.keys()),
                                                   global_step_transform=global_step_from_engine(trainer)),
                         event_name=Events.COMPLETED)

        # Store the best model by validation accuracy:
        common.save_best_model_by_val_score(output_path, evaluator, model=model, metric_name='accuracy', n_saved=3,
                                            trainer=trainer, tag="test")

        if config['log_model_grads_every'] is not None:
            tb_logger.attach(trainer,
                             log_handler=GradsHistHandler(model, tag=model.__class__.__name__),
                             event_name=Events.ITERATION_COMPLETED(every=config['log_model_grads_every']))

    if config['crash_iteration'] is not None:
        @trainer.on(Events.ITERATION_STARTED(once=config['crash_iteration']))
        def _(engine):
            raise Exception("STOP at iteration: {}".format(engine.state.iteration))

    resume_from = config['resume_from']
    if resume_from is not None:
        checkpoint_fp = Path(resume_from)
        assert checkpoint_fp.exists(), "Checkpoint '{}' is not found".format(checkpoint_fp.as_posix())
        print("Resume from a checkpoint: {}".format(checkpoint_fp.as_posix()))
        checkpoint = torch.load(checkpoint_fp.as_posix())
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    try:
        trainer.run(train_loader, max_epochs=config['num_epochs'])
    except Exception as e:
        import traceback
        print(traceback.format_exc())

    if rank == 0:
        tb_logger.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Training a CNN on CIFAR10 dataset")

    parser.add_argument('--network', type=str, default="fastresnet", help="Network to train")

    parser.add_argument('--params', type=str,
                        help='Override default configuration with parameters: '
                             'data_path=/path/to/dataset;batch_size=64;num_workers=12 ...')

    parser.add_argument('--local_rank', type=int, help='Local process rank in distributed computation')

    args = parser.parse_args()
    network_name = args.network

    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    batch_size = 512
    num_epochs = 24
    # Default configuration dictionary
    config = {
        "seed": 12,

        "data_path": "/tmp/cifar10",
        "output_path": "/tmp/cifar10-output",

        "model": network_name,

        "momentum": 0.9,
        "weight_decay": 1e-4,
        "batch_size": batch_size,
        "num_workers": 10,

        "num_epochs": num_epochs,

        "learning_rate": 0.04,
        "num_warmup_epochs": 4,

        "validate_every": 3,

        # distributed settings
        "dist_url": "env://",
        "dist_backend": None,  # if None distributed option is disabled, set to "nccl" to enable

        # Logging:
        "display_iters": True,
        "log_model_grads_every": None,
        "checkpoint_every": 200,

        # Crash/Resume training:
        "resume_from": None,  # Path to checkpoint file .pth
        "crash_iteration": None,
    }

    if args.local_rank is not None:
        config['local_rank'] = args.local_rank
    else:
        config['local_rank'] = 0

    # Override config:
    if args.params is not None:
        for param in args.params.split(";"):
            key, value = param.split("=")
            if "/" not in value:
                value = eval(value)
            config[key] = value

    backend = config['dist_backend']
    distributed = backend is not None

    if distributed:
        dist.init_process_group(backend, init_method=config['dist_url'])
        # let each node print the info
        if config['local_rank'] == 0:
            print("\nDistributed setting:")
            print("\tbackend: {}".format(dist.get_backend()))
            print("\tworld size: {}".format(dist.get_world_size()))
            print("\trank: {}".format(dist.get_rank()))
            print("\n")

    output_path = None
    # let each node print the info
    if config['local_rank'] == 0:
        print("Train {} on CIFAR10".format(network_name))
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

            output_path = Path(config['output_path']) / "{}{}".format(now, gpu_conf)
            if not output_path.exists():
                output_path.mkdir(parents=True)
            output_path = output_path.as_posix()
            print("Output path: {}".format(output_path))

    try:
        run(output_path, config)
    except KeyboardInterrupt:
        print("Catched KeyboardInterrupt -> exit")
    except Exception as e:
        if distributed:
            dist.destroy_process_group()
        raise e

    if distributed:
        dist.destroy_process_group()
