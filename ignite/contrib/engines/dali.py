try:
    from nvidia.dali import pipeline, ops, types
except ImportError:
    raise RuntimeError("This contrib module requires nvidia-dali to be installed")

from ignite.engine import Engine

import torch
from torch.nn.parallel import gather, parallel_apply, replicate


def _prepare_batch(batch,
                   device_ids=None,
                   output_map=('data', 'label')):
    outputs = [(b[o] for o in output_map) for b in batch]
    return zip(*outputs)


def create_supervised_dali_evaluator(model,
                                     metrics,
                                     device_ids=None,
                                     output_map=["data", "label"],
                                     output_device=-1,
                                     prepare_batch=_prepare_batch):

    replicas = model.cuda()
    if device_ids:
        replicas = replicate(model, device_ids)

    def _inference(engine, batch):
        for r in replicas:
            r.eval()
        x, y = prepare_batch(batch,
                             device_ids=device_ids,
                             output_map=output_map)
        y_pred = parallel_apply(replicas[:len(x)], x)
        y_pred = gather(y_pred, output_device, 0)
        if y[0].device.type == 'cpu':
            y = torch.cat(y)
        else:
            y = gather(y, output_device, 0)
        return y_pred, y

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_supervised_dali_trainer(model,
                                   optimizer,
                                   loss_fn,
                                   device_ids=None,
                                   output_map=["data", "label"],
                                   output_device=-1,
                                   prepare_batch=_prepare_batch):

    replicas = model.cuda()
    if device_ids:
        replicas = replicate(model, device_ids)

    def _update(engine, batch):
        for r in replicas:
            r.train()
        x, y = prepare_batch(batch,
                             device_ids=device_ids,
                             output_map=output_map)
        y_pred = parallel_apply(replicas[:len(x)], x)
        y_pred = gather(y_pred, output_device, 0)
        if y[0].device.type == 'cpu':
            y = torch.cat(y)
        else:
            y = gather(y, output_device, 0)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)

def create_unsupervised_dali_evaluator(model,
                                       device_ids=None,
                                       output_map=["data"],
                                       output_device=-1,
                                       prepare_batch=_prepare_batch):

    replicas = model.cuda()
    if device_ids:
        replicas = replicate(model, device_ids)

    def _inference(engine, batch):
        for r in replicas:
            r.eval()
        x = prepare_batch(batch,
                          device_ids=device_ids,
                          output_map=output_map)
        y_pred = parallel_apply(replicas[:len(x)], x)
        y_pred = gather(y_pred, output_device, 0)
        return y_pred

    return Engine(_inference)
