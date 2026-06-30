import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali import pipeline
import nvidia.dali.ops as ops
from typing import Sequence


def _pipelines_sizes(pipes):
    for p in pipes:
        p.build()
        keys = list(p.epoch_size().keys())
        if len(keys) > 0:
            for k in keys:
                yield p.epoch_size(k)
        else:
            yield len(p)


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


class DALILoader(DALIGenericIterator):
    """
    Class to make a `DALIGenericIterator` because `ProgressBar` wants an object with a
    `__len__` method. Also the `ProgressBar` is updated by step of 1 !
    """

    def __init__(self, pipelines, output_map=("data", "label"), auto_reset=True, stop_at_epoch=True):
        if not isinstance(pipelines, Sequence):
            pipelines = [pipelines]
        size = sum(_pipelines_sizes(pipelines))
        super().__init__(pipelines, output_map, size, auto_reset, stop_at_epoch)
        self.batch_size = pipelines[0].batch_size

    def __len__(self):
        return self._size // self.batch_size


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
        # TODO: Take into account the cpu case
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
        targets = self.labels
        if self.transform:
            self.jpegs = self.transform(self.jpegs)
        if self.target_transform:
            targets = self.target_transform(targets)
        return self.jpegs, targets

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
                """a `Pipeline` expect a list of `batch_size` elements but at the end of
                an epoch this size is not guaranted so we repeat the last element.
                However the `Pipeline` return `Pipeline.size` elements even if `Pipeline.size`
                is not a multiple of `batch_size`"""
                s = self.samples[-1]
                samples = chain.from_iterable([samples, repeat(s, diff)])

            jpegs, labels = self._iter_setup(samples)
            self.feed_input(self.jpegs, jpegs)
            self.feed_input(self.labels, labels)
            self.slice = slice(sl.stop, sl.stop + self.batch_size, sl.step)

    def reset(self):
        if self._iter_setup:
            self.slice = slice(0, self.batch_size)
