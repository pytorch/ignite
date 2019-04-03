from argparse import ArgumentParser
from os import path
from itertools import chain, repeat
from math import ceil
from random import sample

import numpy as np

import torch
from torch import nn, multiprocessing
from torch.optim import SGD
import torch.nn.functional as F
from torch.nn.parallel import replicate

from torchvision.models import resnet18
from torchvision.datasets import ImageFolder

from nvidia.dali import pipeline, ops, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from ignite.engine import Events
from ignite.metrics import Loss, RunningAverage, Accuracy

from ignite.contrib.handlers import ProgressBar
from ignite.contrib.engines import (create_supervised_dali_trainer,
                                    create_supervised_dali_evaluator)


def read_from_paths(paths, dtype=np.uint8):
    """
    Read the bytes from a path
    """
    for p in paths:
        with open(p, 'rb') as src:
            img = np.frombuffer(src.read(), dtype=dtype)
        yield img


def finetune_model(model, out_features, freeze=True):
    """
    Replace last linear layer with a new one
    """
    in_features = model.fc.in_features
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(in_features, out_features)
    nn.init.xavier_uniform_(model.fc.weight)
    return model


def trainval_split(samples):
    samples = set(samples)
    val_samples = set(sample(samples, int(len(samples)*val_ratio)))
    train_samples = samples - val_samples
    return train_samples, val_samples


class DALILoader(DALIGenericIterator):
    """
    Class to make a `DALIGenericIterator` because `ProgressBar` wants an object with a
    `__len__` method. Also the `ProgressBar` is updated by step of 1 !
    """

    def __init__(self,
                 pipelines,
                 output_map,
                 size,
                 auto_reset=False,
                 stop_at_epoch=False,
                 iter_size=1):
        super().__init__(pipelines, output_map, size, auto_reset, stop_at_epoch)
        self.iter_size = iter_size

    def __len__(self):
        return int(ceil(self._size / self.iter_size))


class SamplesPipeline(pipeline.Pipeline):
    def __init__(self,
                 samples,
                 batch_size,
                 crop=224,
                 output_type=types.FLOAT,
                 device_id=0,
                 num_threads=multiprocessing.cpu_count(),
                 randomize=True):
        super().__init__(batch_size,
                         num_threads,
                         device_id,
                         seed=15)
        self.randomize = randomize
        if randomize:
            samples = sample(samples, k=len(samples))
        self.samples = samples
        self.input_jpegs = ops.ExternalSource()
        self.input_labels = ops.ExternalSource()
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

        self.resize = ops.Resize(device='gpu', resize_x=crop, resize_y=crop)
        self.np = ops.NormalizePermute(device='gpu',
                                       height=crop,
                                       width=crop,
                                       mean=mean,
                                       std=std,
                                       image_type=types.RGB,
                                       output_dtype=output_type)
        self.slice = slice(0, self.batch_size)

    def define_graph(self):
        self.jpegs = self.input_jpegs()
        self.labels = self.input_labels()
        images = self.decode(self.jpegs)
        resized = self.resize(images)
        outputs = self.np(resized)
        return outputs, self.labels

    def iter_setup(self):
        sl = self.slice
        samples = self.samples[sl]
        diff = self.batch_size - len(samples)
        if diff > 0:
            """
            Complete the last batch with the last sample because all batches must
            have the same size
            """
            s = self.samples[-1]
            samples = chain.from_iterable([samples, repeat(s, diff)])
        paths, categories = zip(*samples)
        inputs = list(read_from_paths(paths))
        # `Pipeline.feed_input` seems to only accept `list[np.ndarray]` as input
        categories = [np.array([c], dtype=np.uint8) for c in categories]

        self.feed_input(self.jpegs, inputs)
        self.feed_input(self.labels, categories)
        self.slice = slice(sl.stop, sl.stop+self.batch_size, sl.step)

    def reset(self):
        super().reset()
        self.samples = sample(self.samples, k=len(self.samples))
        self.slice = slice(0, self.batch_size)

    def __len__(self):
        return len(self.samples)


def make_pipelines(samples,
                   crop,
                   batch_size,
                   num_gpus,
                   output_type=types.FLOAT):

    pipelines = []
    samples = tuple(samples)
    for i in range(num_gpus):
        pipe = SamplesPipeline(samples[i::num_gpus], batch_size, crop, output_type, i)
        pipelines.append(pipe)
    return pipelines


def run(file_root,
        train_batch_size,
        val_batch_size,
        epochs,
        lr,
        momentum,
        num_gpus,
        val_ratio,
        crop):

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    if val_batch_size is None:
        val_batch_size = train_batch_size

    dataset = ImageFolder(file_root)
    mapping = dataset.class_to_idx

    train_pipelines = make_pipelines(train_samples, crop, train_batch_size, num_gpus)

    train_loader = DALILoader(
        train_pipelines,
        output_map=['data', 'label'],
        size=sum(len(t) for t in train_pipelines),
        stop_at_epoch=True,
        auto_reset=True,
        iter_size=train_batch_size * num_gpus
    )

    model = resnet18(pretrained=True)
    model = finetune_model(model, len(mapping.keys()), freeze=True)

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

    train_ids = [p.device_id for p in train_pipelines]
    models = replicate(model.cuda(), train_ids)

    def loss_fn(inputs, targets):
        return F.cross_entropy(inputs, targets.view(-1).long())
    loss = Loss(loss_fn)
    accuracy = Accuracy(output_transform=lambda x: (x[0], x[1].view(-1).long()))

    trainer = create_supervised_dali_trainer(models,
                                             optimizer,
                                             loss_fn,
                                             device=None)
    evaluator = create_supervised_dali_evaluator(models,
                                                 metrics={
                                                     'loss': loss,
                                                     'accuracy': accuracy,
                                                 },
                                                 device=None)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, ['loss'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['loss']
        pbar.log_message(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        val_size = val_batch_size * num_gpus
        val_pipelines = make_pipelines(val_samples, crop, val_batch_size, num_gpus)
        val_loader = DALILoader(
            val_pipelines,
            output_map=['data', 'label'],
            size=sum(len(v) for v in val_pipelines),
            auto_reset=True,
            stop_at_epoch=True,
            iter_size=val_size
        )

        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['loss']
        pbar.log_message(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))

        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=epochs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--file_root', type=str,
                        help='Path to the root of the dataset')
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
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of gpus to use')
    parser.add_argument('--val_ratio', type=float, default=.3,
                        help='ratio of images to use for validation')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='ratio of images to use for validation')

    args = parser.parse_args()

    run(
        args.file_root,
        args.batch_size,
        args.val_batch_size,
        args.epochs,
        args.lr,
        args.momentum,
        args.num_gpus,
        args.val_ratio,
        args.crop_size
    )
