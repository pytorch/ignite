import argparse
from pathlib import Path

from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp

import ignite
from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.metrics import Accuracy, RunningAverage, Loss
from ignite.utils import convert_tensor

from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import OutputHandler as tbOutputHandler, \
    OptimizerParamsHandler as tbOptimizerParamsHandler

from ignite.contrib.handlers import create_lr_scheduler_with_warmup

from utils import set_seed, get_train_test_loaders, get_model


def run(output_path, config):
    device = "cuda"
    batch_size = config['batch_size']
    local_rank = config['local_rank']

    distributed = backend is not None
    train_sampler = None

    if distributed:
        train_sampler = 'distributed'
        torch.cuda.device(config['local_rank'])
        device = "cuda:{}".format(config['local_rank'])
        print("local rank={}: device={}".format(config['local_rank'], device))

    train_labelled_loader, test_loader = \
        get_train_test_loaders(path=config['data_path'],
                               batch_size=batch_size,
                               train_sampler=train_sampler,
                               num_workers=config['num_workers'])

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
    num_train_steps = le * config['num_epochs']

    lr = config['learning_rate']
    eta_min = lr * config['min_lr_ratio']
    num_warmup_steps = config['num_warmup_steps']

    lr_scheduler = CosineAnnealingLR(optimizer, eta_min=eta_min, T_max=num_train_steps - num_warmup_steps)

    if num_warmup_steps > 0:
        lr_scheduler = create_lr_scheduler_with_warmup(lr_scheduler,
                                                       warmup_start_value=0.0,
                                                       warmup_end_value=lr * (1.0 + 1.0 / num_warmup_steps),
                                                       warmup_duration=num_warmup_steps)

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

    if local_rank == 0:
        metric_names = [
            'batch loss',
        ]

        def output_transform(x, name):
            return x[name]

        for n in metric_names:
            RunningAverage(output_transform=partial(output_transform, name=n), epoch_bound=False).attach(trainer, n)

        ProgressBar(persist=True, bar_format="").attach(trainer,
                                                        event_name=Events.EPOCH_STARTED,
                                                        closing_event_name=Events.COMPLETED)

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
        "accuracy": Accuracy(),
        "loss": Loss(criterion)
    }

    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)

    def run_validation(engine, val_interval):
        if engine.state.epoch % val_interval == 0:
            train_evaluator.run(train_labelled_loader)
            evaluator.run(test_loader)

    trainer.add_event_handler(Events.EPOCH_STARTED, run_validation, val_interval=3)
    trainer.add_event_handler(Events.COMPLETED, run_validation, val_interval=1)

    if local_rank == 0:
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

    trainer.run(train_labelled_loader, max_epochs=config['num_epochs'])


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

    batch_size = 128
    num_epochs = 100
    config = {
        "data_path": ".",
        "output_path": "output",

        "model": network_name,

        "momentum": 0.9,
        "weight_decay": 1e-4,
        "batch_size": batch_size,
        "num_workers": 10,

        "num_epochs": num_epochs,

        "learning_rate": 0.03,
        "min_lr_ratio": 0.004,
        "num_warmup_steps": 0,

        # distributed settings
        "dist_url": "env://",
        "dist_backend": None,  # if None distributed option is disabled, set to "nccl" to enable
    }

    if args.local_rank is not None:
        config['local_rank'] = args.local_rank

    # Override config:
    if args.params is not None:
        for param in args.params.split(";"):
            key, value = param.split("=")
            if "/" not in value:
                value = eval(value)
            config[key] = value

    output_path = None
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

        from datetime import datetime

        output_path = Path(config['output_path']) / "{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
        if not output_path.exists():
            output_path.mkdir(parents=True)
        output_path = output_path.as_posix()
        print("Output path: {}".format(output_path))

    try:
        backend = config['dist_backend']
        distributed = backend is not None
        if distributed:
            dist.init_process_group(backend, init_method=config['dist_url'])

        run(output_path, config)

    except KeyboardInterrupt:
        dist.destroy_process_group()
