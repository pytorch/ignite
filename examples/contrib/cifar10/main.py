import argparse
from pathlib import Path

from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.parallel
import torch.distributed as dist

import ignite
from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.metrics import Accuracy, RunningAverage, Loss
from ignite.handlers import ModelCheckpoint
from ignite.utils import convert_tensor

from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import OutputHandler as tbOutputHandler, \
    OptimizerParamsHandler as tbOptimizerParamsHandler

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

    # Rescale batch_size and num_workers
    ngpus_per_node = torch.cuda.device_count()
    ngpus = dist.get_world_size() if distributed else 1
    batch_size = config['batch_size'] // ngpus
    num_workers = int((config['num_workers'] + ngpus_per_node - 1) / ngpus_per_node)

    train_labelled_loader, test_loader = \
        get_train_test_loaders(path=config['data_path'],
                               batch_size=batch_size,
                               distributed=distributed,
                               num_workers=num_workers)

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

    le = len(train_labelled_loader)
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

    def process_function(engine, labelled_batch):

        x, y = _prepare_batch(labelled_batch, device=device, non_blocking=True)

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

    if not hasattr(lr_scheduler, "step"):
        trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)
    else:
        trainer.add_event_handler(Events.ITERATION_STARTED, lambda engine: lr_scheduler.step())

    metric_names = [
        'batch loss',
    ]

    def output_transform(x, name):
        return x[name]

    for n in metric_names:
        # We compute running average values on the output (batch loss) across all devices
        RunningAverage(output_transform=partial(output_transform, name=n),
                       epoch_bound=False, device=device).attach(trainer, n)

    if rank == 0:
        checkpoint_handler = ModelCheckpoint(dirname=output_path,
                                             filename_prefix="checkpoint",
                                             save_interval=1000)
        trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                  checkpoint_handler,
                                  {'model': model, 'optimizer': optimizer})

        ProgressBar(persist=True, bar_format="").attach(trainer,
                                                        event_name=Events.EPOCH_STARTED,
                                                        closing_event_name=Events.COMPLETED)
        if config['display_iters']:
            ProgressBar(persist=False, bar_format="").attach(trainer, metric_names=metric_names)

        tb_logger = TensorboardLogger(log_dir=output_path)
        tb_logger.attach(trainer,
                         log_handler=tbOutputHandler(tag="train",
                                                     metric_names=metric_names),
                         event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer,
                         log_handler=tbOptimizerParamsHandler(optimizer, param_name="lr"),
                         event_name=Events.ITERATION_STARTED)

    metrics = {
        "accuracy": Accuracy(device=device if distributed else None),
        "loss": Loss(criterion, device=device if distributed else None)
    }

    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)

    def run_validation(engine):
        torch.cuda.synchronize()
        train_evaluator.run(train_labelled_loader)
        evaluator.run(test_loader)

    trainer.add_event_handler(Events.EPOCH_STARTED(every=3), run_validation)
    trainer.add_event_handler(Events.COMPLETED, run_validation)

    if rank == 0:
        if config['display_iters']:
            ProgressBar(persist=False, desc="Train evaluation").attach(train_evaluator)
            ProgressBar(persist=False, desc="Test evaluation").attach(evaluator)

        tb_logger.attach(train_evaluator,
                         log_handler=tbOutputHandler(tag="train",
                                                     metric_names=list(metrics.keys()),
                                                     another_engine=trainer),
                         event_name=Events.COMPLETED)

        tb_logger.attach(evaluator,
                         log_handler=tbOutputHandler(tag="test",
                                                     metric_names=list(metrics.keys()),
                                                     another_engine=trainer),
                         event_name=Events.COMPLETED)

        # Store the best model
        def default_score_fn(engine):
            score = engine.state.metrics['accuracy']
            return score

        score_function = default_score_fn if not hasattr(config, "score_function") else config.score_function

        best_model_handler = ModelCheckpoint(dirname=output_path,
                                             filename_prefix="best",
                                             n_saved=3,
                                             score_name="val_accuracy",
                                             score_function=score_function)
        evaluator.add_event_handler(Events.COMPLETED, best_model_handler, {'model': model, })

    trainer.run(train_labelled_loader, max_epochs=config['num_epochs'])

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
    config = {
        "data_path": ".",
        "output_path": "output",

        "model": network_name,

        "momentum": 0.9,
        "weight_decay": 1e-4,
        "batch_size": batch_size,
        "num_workers": 10,

        "num_epochs": num_epochs,

        "learning_rate": 0.04,
        "num_warmup_epochs": 4,

        # distributed settings
        "dist_url": "env://",
        "dist_backend": None,  # if None distributed option is disabled, set to "nccl" to enable

        # Logging:
        "display_iters": True
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

    if distributed:
        dist.destroy_process_group()
