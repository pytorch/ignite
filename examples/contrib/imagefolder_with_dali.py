import os
import inspect
from argparse import ArgumentParser
from itertools import chain, repeat
from math import ceil
from random import sample
from collections import defaultdict
from itertools import repeat, chain
from typing import Sequence
from pathlib import Path

import numpy as np

import torch
from torch import nn, multiprocessing
from torch.optim import SGD
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision.models
from torchvision.datasets import ImageFolder

from nvidia.dali import pipeline, ops, types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator

from ignite.engine import Events
from ignite.metrics import Loss, RunningAverage, Accuracy

from ignite.contrib.handlers import ProgressBar
from ignite.contrib.engines import create_supervised_dali_trainer, create_supervised_dali_evaluator, reduce_tensor, ComposeOps, TransformPipeline
from ignite.metrics import Metric


MODELS = dict(inspect.getmembers(torchvision.models, inspect.isfunction))

def iter_setup(samples):
    """
    Return a (Sequence[np.ndarray], Sequence[np.ndarray])
    """
    paths, labels = zip(*samples)
    def read_path(p):
        with open(p, 'rb') as src:
            return np.frombuffer(src.read(), dtype=np.uint8)

    def read_label(l):
        #WARNING: Becareful not using [] here result in a silent quit of the script
        return np.array([l], dtype=np.uint8)

    jpegs = [read_path(p) for p in paths]
    targets = [read_label(l) for l in labels]
    return jpegs, targets


def make_samples(root, local_rank, world_size, val_ratio):
    dataset = ImageFolder(root)
    mapping = dataset.class_to_idx

    sample_by_category = defaultdict(list)

    for v, k in dataset.samples:
        sample_by_category[k].append((str(v), k))

    def gen_samples(sample_by_category):
        for v in sample_by_category.values():
            yield v[local_rank::world_size]

    samples = tuple(chain.from_iterable(gen_samples(sample_by_category)))
    train, val = trainval_split(samples, 1-val_ratio)
    return train, val, len(sample_by_category.keys())

def prepare_batch(batch, device, output_map):
    x = batch[0]['data']
    y = batch[0]['label']
    y = y.squeeze().long().to('cuda')
    return x, y


def finetune_model(model, out_features, requires_grad=False):
    """
    Replace last linear layer with a new one
    """

    name = model.__class__.__name__

    for param in model.parameters():
        param.requires_grad = requires_grad

    # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    if name == 'ResNet':
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, out_features)
    elif name == 'Alexnet':
        model.classifier[6] = nn.Linear(512, out_features)
    elif name == 'VGG':
        model.classifier[6] = nn.Linear(4096, out_features)
    elif name == 'Squeezenet':
        model.classifier[1] = nn.Conv2d(512,
                                        out_features,
                                        kernel_size=(1,1),
                                        stride=(1,1))
    elif name == 'DenseNet':
        model.classifier = nn.Linear(1024, out_features)
    elif name == 'Inception3':
        model.AuxLogits.fc = nn.Linear(768, out_features)
        model.fc = nn.Linear(2048, out_features)
    else:
        raise Exception("Invalid model name {}".format(name))

    return model


def make_model(name, n_categories, pretrained=True, requires_grad=False):
    model = MODELS[name](pretrained=pretrained)
    input_size = 224
    name = model.__class__.__name__

    if name == 'Inception3':
        input_size = 229

    model = finetune_model(model, n_categories, requires_grad)
    return model, input_size


def trainval_split(samples, train_ratio):

    sample_by_category = defaultdict(list)
    for v, k in samples:
        sample_by_category[k].append((v, k))

    def split(values, train_ratio=train_ratio):
        train_values = set(sample(values, int(len(values)*train_ratio)))
        val_values = set(values) - set(train_values)
        return tuple(train_values), tuple(val_values)

    train_samples, val_samples = zip(*(split(v) for  v in sample_by_category.values()))

    return tuple(chain.from_iterable(train_samples)), tuple(chain.from_iterable(val_samples))


def _pipelines_sizes(pipes):
    for p in pipes:
        p.build()
        keys = list(p.epoch_size().keys())
        if len(keys) > 0:
            for k in keys:
                yield p.epoch_size(k)
        else:
            yield len(p)


class DALILoader(DALIGenericIterator):
    """
    Class to make a `DALIGenericIterator` because `ProgressBar` wants an object with a
    `__len__` method. Also the `ProgressBar` is updated by step of 1 !
    """

    def __init__(self,
                 pipelines,
                 output_map,
                 auto_reset=False,
                 stop_at_epoch=False):
        if not isinstance(pipelines, Sequence):
            pipelines = [pipelines]
        size = sum(_pipelines_sizes(pipelines))
        super().__init__(pipelines, output_map, size, auto_reset, stop_at_epoch)
        self.batch_size = pipelines[0].batch_size

    def __len__(self):
        return int(ceil(self._size /self.batch_size ))


def run(model_name,
        root,
        train_batch_size,
        val_batch_size,
        epochs,
        lr,
        momentum,
        val_ratio,
        requires_grad,
        pretrained,
        local_rank):

    world_size = int(os.environ.get('WORLD_SIZE', 1))

    device_id = local_rank % torch.cuda.device_count()

    torch.cuda.set_device(device_id)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://',
                                         world_size=world_size,
                                         rank=local_rank)
    world_size = torch.distributed.get_world_size()
    train_samples, val_samples, n_categories = make_samples(root,
                                                            local_rank,
                                                            world_size,
                                                            val_ratio)
    model, input_size = make_model(model_name,
                                   n_categories,
                                   requires_grad=requires_grad,
                                   pretrained=pretrained)
    torch.cuda.set_device(device_id)
    model.to('cuda')
    model = DDP(model, device_ids=[device_id], output_device=device_id)

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

    loss_fn = F.cross_entropy
    loss = Loss(lambda y_pred, y: reduce_tensor(loss_fn(y_pred, y), world_size))
    # Can't be used in a distributed as it is. Must write a class DistributedMetrics
    # accuracy = Accuracy(output_transform=lambda x, y: ())
    # Values from imagenet preprocessing
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    transform = ComposeOps([
        ops.RandomResizedCrop(device='gpu', size=(input_size, input_size)),
        ops.CropMirrorNormalize(device="gpu",
                                output_dtype=types.FLOAT,
                                output_layout=types.NCHW,
                                crop=(input_size, input_size),
                                image_type=types.RGB,
                                mean=mean,
                                std=std)
    ])

    pipe = TransformPipeline(batch_size=train_batch_size,
                             samples=train_samples,
                             num_threads=8,
                             device_id=device_id,
                             transform=transform,
                             size=len(train_samples),
                             iter_setup=iter_setup)

    train_loader = DALILoader(pipe,
                              ['data', 'label'],
                              auto_reset=True,
                              stop_at_epoch=True)

    trainer = create_supervised_dali_trainer(model,
                                             optimizer,
                                             loss_fn,
                                             device=None,
                                             prepare_batch=prepare_batch,
                                             world_size=world_size)
    evaluator = create_supervised_dali_evaluator(model,
                                                 metrics={
                                                     'loss': loss,
                                                     # 'accuracy': accuracy,
                                                 },
                                                 device=None,
                                                 world_size=world_size,
                                                 prepare_batch=prepare_batch)

    if local_rank == 0:
        RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, ['loss'])

    if val_batch_size is None:
        val_batch_size = train_batch_size

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        # avg_accuracy = metrics['accuracy']
        if local_rank == 0:
            metrics = evaluator.state.metrics
            # accumulated loss is shared between all processes but the number of samples is not the same per process so the average loss is slightly diferrent between the process but it converge to the same limit
            avg_loss = metrics['loss']
            # pbar.log_message(
            #     "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            #     .format(engine.state.epoch, avg_accuracy, avg_loss)
            # )
            pbar.log_message(
                "Training Results - Epoch: {} Avg loss: {:.2f}"
                .format(engine.state.epoch, avg_loss)
            )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_val_results(engine):

        pipe = TransformPipeline(
            batch_size=val_batch_size,
            samples=val_samples,
            num_threads=8,
            device_id=device_id,
            transform=transform,
            size=len(val_samples),
            iter_setup=iter_setup)

        val_loader = DALILoader(
            pipe,
            ['data', 'label'],
            auto_reset=True,
            stop_at_epoch=True
        )

        evaluator.run(val_loader)
        if local_rank == 0:
            metrics = evaluator.state.metrics
            # avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']
            # pbar.log_message(
            #     "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            #     .format(engine.state.epoch, avg_accuracy, avg_loss)
            # )
            pbar.log_message(
                "Validation Results - Epoch: {} Avg loss: {:.2f}"
                .format(engine.state.epoch, avg_loss)
            )

    trainer.run(train_loader, max_epochs=epochs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--root', type=str,
                        help='Path to the root of the dataset')
    names = '|'.join(MODELS.keys())
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Name of the model in <{}>'.format(names))
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=None,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--val_ratio', type=float, default=.3,
                        help='ratio of images to use for validation')
    parser.add_argument('--requires_grad', type=bool, default=False,
                        help='Finetune model')
    parser.add_argument('--pretrained', type=bool, default=True,
                        help='Use pretrained model')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank')

    args = parser.parse_args()

    run(
        args.model,
        args.root,
        args.batch_size,
        args.val_batch_size,
        args.epochs,
        args.lr,
        args.momentum,
        args.val_ratio,
        args.requires_grad,
        args.pretrained,
        args.local_rank,
    )
