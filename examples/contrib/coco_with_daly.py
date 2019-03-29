from argparse import ArgumentParser
from os import path
from glob import glob
from itertools import chain, repeat
from collections import defaultdict
from types import MethodType

import numpy as np

import torch
from torch import nn, multiprocessing
from torch.optim import SGD
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torchvision.models import resnet18

from pycocotools.coco import COCO

from nvidia.dali import pipeline, ops, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from ignite.engine import Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.utils import to_onehot

from ignite.contrib.handlers import ProgressBar
from ignite.contrib.engines import (create_supervised_dali_trainer,
                                    create_supervised_dali_evaluator)


def read_from_paths(root, paths, dtype=np.uint8):
    """
    Read the bytes from a path
    """

    for p in paths:
        with open(path.join(root, p), 'rb') as src:
            img = np.frombuffer(src.read(), dtype=dtype)
        yield img


def make_resources(annotations_file, device_id, num_gpus):
    """Get paths and classes"""
    coco = COCO(annotations_file)
    cats = coco.cats.keys()
    categories = dict(enumerate(cats))
    reverse_cat = dict((c, i) for i, c in categories.items())

    all_ids = [*coco.imgs.keys()]
    img_ids = all_ids[device_id::num_gpus]
    ann_ids = coco.getAnnIds(imgIds=img_ids)
    anns = coco.loadAnns(ann_ids)
    targets = defaultdict(list)
    paths = dict((i['id'], i['file_name']) for i in coco.loadImgs(img_ids))

    for a in anns:
        targets[a['image_id']].append(reverse_cat[a['category_id']])

    return (tuple((paths[img_id], list(targets[img_id])) for img_id in img_ids),
            categories)


class DALILoader(DALIGenericIterator):
    """
    Class to make a `DALIGenericIterator` because `ProgressBar` want an object with a
    `__len__` method
    """

    def __init(self, pipelines, output_map, size, auto_reset=True, stop_at_epoch=False):
        super().__init__(pipelines, output_map, size, auto_reset, stop_at_epoch)

    def __len__(self):
        return self._size


class COCOPipeline(pipeline.Pipeline):
    """
    Pipeline for coco with data augrmentation
    """
    def __init__(self,
                 file_root,
                 annotations_file,
                 batch_size,
                 crop=224,
                 output_type=types.FLOAT,
                 device_id=0,
                 num_gpus=1,
                 num_threads=multiprocessing.cpu_count()):
        super(COCOPipeline, self).__init__(batch_size,
                                           num_threads,
                                           device_id,
                                           seed=15)
        self.root = file_root
        samples, categories = make_resources(annotations_file, device_id, num_gpus)
        self.samples = samples
        self.categories = categories
        self.n_categories = len(categories.keys())
        self.input_jpegs = ops.ExternalSource()
        self.input_labels = ops.ExternalSource()
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        # self.cmnp = ops.CropMirrorNormalize(device="gpu",
        #                                     output_dtype=output_type,
        #                                     output_layout=types.NCHW,
        #                                     crop=crop,
        #                                     image_type=types.RGB,
        #                                     mean=mean,
        #                                     std=std)

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
            s = self.samples[-1]
            samples = chain.from_iterable([samples, repeat(s, diff)])
        paths, categories = zip(*samples)
        inputs = list(read_from_paths(self.root, paths))
        targets = np.zeros((len(categories), self.n_categories), dtype=np.int32)
        for i, c in enumerate(categories):
            targets[i, c] = 1

        self.feed_input(self.jpegs, inputs)
        self.feed_input(self.labels, targets)
        self.slice = slice(sl.stop, sl.stop+self.batch_size, sl.step)

    def reset(self):
        self.slice = slice(0, self.batch_size)
        return super().reset()

    def __len__(self):
        return len(self.samples)


def make_pipelines(root, crop, subset, year, batch_size):
    num_gpus = torch.cuda.device_count()
    file_root = path.join(root, f'{subset}{year}')
    annotations_file = path.join(root,
                                 'annotations',
                                 f'instances_{subset}{year}.json')

    pipelines = []
    for i in range(num_gpus):
        pipe = COCOPipeline(file_root=file_root,
                            annotations_file=annotations_file,
                            crop=crop,
                            device_id=i,
                            batch_size=batch_size,
                            num_threads=multiprocessing.cpu_count(),
                            num_gpus=num_gpus)
        pipelines.append(pipe)
    return pipelines



def run(file_root, train_batch_size, val_batch_size, epochs, lr, momentum, year):

    crop = 224
    train_pipelines = make_pipelines(file_root, crop, 'train', year, train_batch_size)

    train_loader = DALILoader(train_pipelines,
                              output_map=['data', 'label'],
                              size=sum(len(p) for p in train_pipelines),
                              stop_at_epoch=True)

    val_pipelines = make_pipelines(file_root, crop, 'val', year, train_batch_size)
    val_loader = DALILoader(val_pipelines,
                            output_map=['data', 'label'],
                            size=sum(len(p) for p in val_pipelines),
                            stop_at_epoch=True)


    model = resnet18(pretrained=False)
    last = model.fc
    model.fc = nn.Linear(last.in_features, len(train_pipelines[0].categories))

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    device_ids = [p.device_id for p in train_pipelines]

    def prepare_batch(batch,
                      device_ids=None,
                      output_map=('data', 'label')):
        return [b['data'] for b in batch], [b['label'].long() for b in batch]

    trainer = create_supervised_dali_trainer(model,
                                             optimizer,
                                             F.multilabel_margin_loss,
                                             device_ids=device_ids,
                                             prepare_batch=prepare_batch)
    loss = Loss(F.multilabel_margin_loss)
    evaluator = create_supervised_dali_evaluator(model,
                                                 metrics={'accuracy': Accuracy(),
                                                          'nll': loss},
                                                 device_ids=device_ids)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, ['loss'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        pbar.log_message(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        pbar.log_message(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))

        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=epochs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--file_root', type=str,
                        help='Path to the root of COCO')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1000,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--year', type=int, default=2014,
                        help='COCO dataset year')

    args = parser.parse_args()

    run(args.file_root,
        args.batch_size,
        args.val_batch_size,
        args.epochs,
        args.lr,
        args.momentum,
        args.year)
