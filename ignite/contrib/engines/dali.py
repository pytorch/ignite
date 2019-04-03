try:
    from nvidia.dali import pipeline, ops, types
except ImportError:
    raise RuntimeError("This contrib module requires nvidia-dali to be installed")

from typing import Sequence

import torch
from torch.nn.parallel import gather, parallel_apply

from ignite.engine import Engine


def _prepare_batch(batch,
                   device=None,
                   output_map=('data', 'label')):
    outputs = [[b[o] for o in output_map] for b in batch]
    return tuple(zip(*outputs))


def _apply_and_gather(models, x, y=None, output_device=-1):
    indexes = [xx.device.index for xx in x]
    models = [models[i] for i in indexes]
    y_pred = parallel_apply(models, x)
    y_pred = gather(y_pred, output_device, 0)
    if y is None:
        return y_pred
    if y[0].device.type == 'cpu':
        y = torch.cat(y)
    else:
        y = gather(y, output_device, 0)

    return y_pred, y


def create_supervised_dali_trainer(models,
                                   optimizer,
                                   loss_fn,
                                   device=None,
                                   output_map=["data", "label"],
                                   output_device=-1,
                                   prepare_batch=_prepare_batch):

    if not isinstance(models, Sequence):
        models = [models]

    def _update(engine, batch, models=models):

        for r in models:
            r.train()

        x, y = prepare_batch(batch,
                             device=device,
                             output_map=output_map)
        y_pred, y = _apply_and_gather(models, x, y, output_device)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    engine = Engine(_update)
    return engine


def create_supervised_dali_evaluator(models,
                                     metrics,
                                     device=None,
                                     output_map=["data", "label"],
                                     output_device=-1,
                                     prepare_batch=_prepare_batch):

    if not isinstance(models, Sequence):
        models = [models]

    def _inference(engine, batch, models=models):
        for r in models:
            r.eval()

        with torch.no_grad():
            x, y = prepare_batch(batch,
                                 device,
                                 output_map=output_map)
            y_pred, y = _apply_and_gather(models, x, y, output_device)
            return y_pred, y

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_unsupervised_dali_evaluator(models,
                                       device=None,
                                       output_map=["data"],
                                       output_device=-1,
                                       prepare_batch=_prepare_batch):

    if not isinstance(models, Sequence):
        models = [models]

    def _inference(engine, batch, models=models):
        for r in models:
            r.eval()

        with torch.no_grad():
            x = prepare_batch(batch,
                              device=device,
                              output_map=output_map)
            y_pred = _apply_and_gather(models, x, output_device)
            return y_pred

    return Engine(_inference)
