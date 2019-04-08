# -*- coding: utf-8 -*-
import numpy as np
import pytest
import torch

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy
from ignite.contrib.handlers import CustomPeriodicEvent


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


def test_pbar_log_message(capsys):

    ProgressBar.log_message("test")

    captured = capsys.readouterr()
    out = captured.out.split('\r')
    out = list(map(lambda x: x.strip(), out))
    out = list(filter(None, out))
    expected = u'test'
    assert out[-1] == expected


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

    accuracy = Accuracy(output_transform=lambda x: (x[1], x[2]))
    accuracy.attach(trainer, "avg_accuracy")

    pbar = ProgressBar()
    pbar.attach(trainer, ['avg_accuracy'])

    with pytest.warns(UserWarning):
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


def test_pbar_with_tqdm_kwargs(capsys):
    n_epochs = 10
    loader = [1, 2, 3, 4, 5]
    engine = Engine(update_fn)

    pbar = ProgressBar(desc="My description: ")
    pbar.attach(engine, output_transform=lambda x: x)
    engine.run(loader, max_epochs=n_epochs)

    captured = capsys.readouterr()
    err = captured.err.split('\r')
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    expected = u'My description:  [10/10]: [4/5]  80%|████████  , output=1.00e+00 [00:00<00:00]'.format()
    assert err[-1] == expected


def test_pbar_for_validation(capsys):
    loader = [1, 2, 3, 4, 5]
    engine = Engine(update_fn)

    pbar = ProgressBar(desc="Validation")
    pbar.attach(engine)
    engine.run(loader, max_epochs=1)

    captured = capsys.readouterr()
    err = captured.err.split('\r')
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    expected = u'Validation: [4/5]  80%|████████   [00:00<00:00]'
    assert err[-1] == expected


def test_pbar_output_tensor(capsys):
    loader = [1, 2, 3, 4, 5]

    def update_fn(engine, batch):
        return torch.Tensor([batch, 0])

    engine = Engine(update_fn)

    pbar = ProgressBar(desc="Output tensor")
    pbar.attach(engine, output_transform=lambda x: x)
    engine.run(loader, max_epochs=1)

    captured = capsys.readouterr()
    err = captured.err.split('\r')
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    expected = u'Output tensor: [4/5]  80%|████████  , output_0=5.00e+00, output_1=0.00e+00 [00:00<00:00]'
    assert err[-1] == expected


def test_pbar_output_warning(capsys):
    loader = [1, 2, 3, 4, 5]

    def update_fn(engine, batch):
        return batch, "abc"

    engine = Engine(update_fn)

    pbar = ProgressBar(desc="Output tensor")
    pbar.attach(engine, output_transform=lambda x: x)
    with pytest.warns(UserWarning):
        engine.run(loader, max_epochs=1)


def test_pbar_on_epochs(capsys):

    n_epochs = 10
    loader = [1, 2, 3, 4, 5]
    engine = Engine(update_fn)

    pbar = ProgressBar()
    pbar.attach(engine, event_name=Events.EPOCH_STARTED, closing_event_name=Events.COMPLETED)
    engine.run(loader, max_epochs=n_epochs)

    captured = capsys.readouterr()
    err = captured.err.split('\r')
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    actual = err[-1]
    expected = u'Epoch: [9/10]  90%|█████████  [00:00<00:00]'
    assert actual == expected


def test_pbar_wrong_events_order():

    engine = Engine(update_fn)
    pbar = ProgressBar()

    with pytest.raises(ValueError, match="should be called before closing event"):
        pbar.attach(engine, event_name=Events.COMPLETED, closing_event_name=Events.COMPLETED)

    with pytest.raises(ValueError, match="should be called before closing event"):
        pbar.attach(engine, event_name=Events.COMPLETED, closing_event_name=Events.EPOCH_COMPLETED)

    with pytest.raises(ValueError, match="should be called before closing event"):
        pbar.attach(engine, event_name=Events.COMPLETED, closing_event_name=Events.ITERATION_COMPLETED)

    with pytest.raises(ValueError, match="should be called before closing event"):
        pbar.attach(engine, event_name=Events.EPOCH_COMPLETED, closing_event_name=Events.EPOCH_COMPLETED)

    with pytest.raises(ValueError, match="should be called before closing event"):
        pbar.attach(engine, event_name=Events.ITERATION_COMPLETED, closing_event_name=Events.ITERATION_STARTED)


def test_pbar_on_custom_events(capsys):

    engine = Engine(update_fn)
    pbar = ProgressBar()
    cpe = CustomPeriodicEvent(n_iterations=15)

    with pytest.raises(ValueError, match=r"Logging and closing events should be only ignite.engine.Events"):
        pbar.attach(engine, event_name=cpe.Events.ITERATIONS_15_COMPLETED, closing_event_name=Events.EPOCH_COMPLETED)
