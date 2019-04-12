try:
    from nvidia.dali import pipeline, ops, types
except ImportError:
    raise RuntimeError("This contrib module requires nvidia-dali to be installed")

from itertools import chain, repeat
from random import sample

import torch
import torch.distributed as dist

from ignite.engine import Engine


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def _prepare_batch(batch, device=None, output_map=("data", "label")):
    outputs = [[b[o] for o in output_map] for b in batch]
    return tuple(zip(*outputs))


class ComposeOps(object):
    """
    Composes several `ops` together
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class TransformPipeline(pipeline.Pipeline):
    """
    Pipeline for coco with data augrmentation
    """

    def __init__(
        self,
        batch_size,
        num_threads,
        device_id,
        size=0,
        transform=None,
        target_transform=None,
        iter_setup=None,
        reader=None,
        samples=None,
        randomize=True,
    ):
        super().__init__(batch_size, num_threads, device_id, seed=-1)
        self.reader = reader
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.transform = transform
        self.target_transform = target_transform
        self._iter_setup = iter_setup
        self.size = size
        self._jpegs = ops.ExternalSource()
        self._labels = ops.ExternalSource()
        self.randomize = randomize
        if randomize and samples:
            samples = sample(samples, len(samples))
        self.samples = samples
        self.slice = slice(0, batch_size)

    def define_graph(self):
        if self.reader is None:
            """
            Default case, jpegs and labels are feed by `ops.ExternalSource`
            """
            self.jpegs = self._jpegs()
            self.labels = self._labels()
        else:
            self.jpegs, self.labels = self.reader()
        outputs = self.decode(self.jpegs)
        targets = self.labels
        if self.transform:
            outputs = self.transform(outputs)
        if self.target_transform:
            targets = self.target_transform(targets)
        return outputs, targets

    def __len__(self):
        keys = list(self.epoch_size().keys())
        if len(keys) > 0:
            size = sum(self.epoch_size(k) for k in keys)
        else:
            size = self.size
        return size

    def iter_setup(self):
        if self._iter_setup:
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

            jpegs, labels = self._iter_setup(samples)
            self.feed_input(self.jpegs, jpegs)
            self.feed_input(self.labels, labels)
            self.slice = slice(sl.stop, sl.stop + self.batch_size, sl.step)

    def reset(self):
        if self._iter_setup:
            self.slice = slice(0, self.batch_size)


def create_supervised_dali_trainer(
    model,
    optimizer,
    loss_fn,
    world_size,
    device=None,
    output_map=("data", "label"),
    prepare_batch=_prepare_batch,
    output_transform=lambda x, y, y_pred, loss: loss.item(),
):
    def _update(engine, batch):

        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, output_map=output_map)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        reduced_loss = reduce_tensor(loss, world_size)
        reduced_loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, reduced_loss)

    engine = Engine(_update)
    return engine


def create_supervised_dali_evaluator(
    model,
    metrics,
    world_size,
    device=None,
    output_map=("data", "label"),
    prepare_batch=_prepare_batch,
    output_transform=lambda x, y, y_pred: (y_pred, y),
):
    def _inference(engine, batch):
        model.eval()

        with torch.no_grad():
            x, y = prepare_batch(batch, device, output_map=output_map)
            y_pred = model(x)
            return output_transform(x, y, y_pred)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
