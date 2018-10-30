# -*- coding: utf-8 -*-
import numpy as np
import pytest
import torch

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine
from ignite.metrics import CategoricalAccuracy


def update_fn(engine, batch):
    a = 1
    engine.state.metrics['a'] = a
    return a


def test_pbar(capsys):

    n_epochs = 2
    loader = [1, 2]
    engine = Engine(update_fn)

    pbar = ProgressBar()
    pbar.attach(engine, ['a'])

    engine.run(loader, max_epochs=n_epochs)

    captured = capsys.readouterr()
    err = captured.err.split('\r')
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    expected = u'Epoch [2/2]: [1/2]  50%|█████     , a=1.00e+00 [00:00<00:00]'
    assert err[-1] == expected


def test_attach_fail_with_string():
    engine = Engine(update_fn)
    pbar = ProgressBar()

    with pytest.raises(TypeError):
        pbar.attach(engine, 'a')


def test_pbar_with_metric():

    n_iters = 20
    batch_size = 10
    n_classes = 2
    data = list(range(n_iters))
    y_true_batch_values = iter(np.random.randint(0, n_classes, size=(n_iters, batch_size)))
    y_pred_batch_values = iter(np.random.rand(n_iters, batch_size, n_classes))
    loss_values = iter(range(n_iters))

    def step(engine, batch):
        loss_value = next(loss_values)
        y_true_batch = next(y_true_batch_values)
        y_pred_batch = next(y_pred_batch_values)
        return loss_value, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch)

    trainer = Engine(step)

    accuracy = CategoricalAccuracy(output_transform=lambda x: (x[1], x[2]))
    accuracy.attach(trainer, "avg_accuracy")

    pbar = ProgressBar()
    pbar.attach(trainer, ['avg_accuracy'])

    with pytest.raises(KeyError):
        trainer.run(data=data, max_epochs=1)


def test_pbar_no_metric_names(capsys):

    n_epochs = 2
    loader = [1, 2]
    engine = Engine(update_fn)

    pbar = ProgressBar()
    pbar.attach(engine)

    engine.run(loader, max_epochs=n_epochs)

    captured = capsys.readouterr()
    err = captured.err.split('\r')
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    actual = err[-1]
    expected = u'Epoch [2/2]: [1/2]  50%|█████      [00:00<00:00]'
    assert actual == expected


def test_pbar_with_output(capsys):
    n_epochs = 2
    loader = [1, 2]
    engine = Engine(update_fn)

    pbar = ProgressBar()
    pbar.attach(engine, output_transform=lambda x: {'a': x})

    engine.run(loader, max_epochs=n_epochs)

    captured = capsys.readouterr()
    err = captured.err.split('\r')
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    expected = u'Epoch [2/2]: [1/2]  50%|█████     , a=1.00e+00 [00:00<00:00]'
    assert err[-1] == expected


def test_pbar_fail_with_non_callable_transform():
    engine = Engine(update_fn)
    pbar = ProgressBar()

    with pytest.raises(TypeError):
        pbar.attach(engine, output_transform=1)


def test_pbar_with_scalar_output(capsys):
    n_epochs = 2
    loader = [1, 2]
    engine = Engine(update_fn)

    pbar = ProgressBar()
    pbar.attach(engine, output_transform=lambda x: x)

    engine.run(loader, max_epochs=n_epochs)

    captured = capsys.readouterr()
    err = captured.err.split('\r')
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    expected = u'Epoch [2/2]: [1/2]  50%|█████     , output=1.00e+00 [00:00<00:00]'
    assert err[-1] == expected
