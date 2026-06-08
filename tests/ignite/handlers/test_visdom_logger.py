# -*- coding: utf-8 -*-
import sys
import time
from argparse import Namespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from packaging.version import Version

from ignite.engine import Engine, Events

from ignite.handlers import ProgressBar, TerminateOnNan
from ignite.metrics import RunningAverage

if sys.platform.startswith("win"):
    pytest.skip("Skip on Windows", allow_module_level=True)


def get_tqdm_version():
    import tqdm

    return Version(tqdm.__version__)


def update_fn(engine, batch):
    a = 1
    engine.state.metrics["a"] = a
    return a


def test_pbar_errors():
    with pytest.raises(ModuleNotFoundError, match=r"This contrib module requires tqdm to be installed"):
        with patch.dict("sys.modules", {"tqdm.autonotebook": None}):
            ProgressBar(ncols=80)

    pbar = ProgressBar(ncols=80)
    with pytest.raises(ValueError, match=r"Logging event abc is not in allowed"):
        pbar.attach(Engine(lambda e, b: None), event_name=Namespace(name="abc"))


def test_pbar(capsys):
    n_epochs = 2
    loader = [1, 2]
    engine = Engine(update_fn)

    pbar = ProgressBar(ncols=80)
    pbar.attach(engine, ["a"])

    engine.run(loader, max_epochs=n_epochs)

    captured = capsys.readouterr()
    err = captured.err.split("\r")
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    if get_tqdm_version() < Version("4.49.0"):
        expected = "Epoch 8 -*-     , a=1 [00:00<00:00]"
    else:
        expected = "Epoch [2/2]: [1/2]  50%|████████████████████▌                    , a=1 [00:00<?]"
    assert err[-1] == expected


def test_pbar_file(tmp_path):
    n_epochs = 2
    loader = [1, 2]
    engine = Engine(update_fn)

    file_path = tmp_path / "temp.txt"
    file = open(str(file_path), "w+")

    pbar = ProgressBar(file=file, ncols=80)
    pbar.attach(engine, ["a"])
    engine.run(loader, max_epochs=n_epochs)

    file.close()  # Force a flush of the buffer. file.flush() does not work.

    file = open(str(file_path), "r")
    lines = file.readlines()

    if get_tqdm_version() < Version("4.49.0"):
        expected = "Epoch [2/2]: [1/2]  50%|█████     , a=1 [00:00<00:00]\n"
    else:
        expected = "Epoch [2/2]: [1/2]  50%|████████████████████▌                    , a=1 [00:00<?]\n"
    assert lines[-2] == expected


def test_pbar_log_message(capsys):
    pbar = ProgressBar(ncols=80)

    pbar.log_message("test")

    captured = capsys.readouterr()
    out = captured.out.split("\r")
    out = list(map(lambda x: x.strip(), out))
    out = list(filter(None, out))
    expected = "test"
    assert out[-1] == expected


def test_pbar_log_message_file(tmp_path):
    file_path = tmp_path / "temp.txt"
    file = open(str(file_path), "w+")

    pbar = ProgressBar(file=file, ncols=80)
    pbar.log_message("test")

    file.close()  # Force a flush of the buffer. file.flush() does not work.

    file = open(str(file_path), "r")
    lines = file.readlines()

    expected = "test\n"
    assert lines[0] == expected


def test_attach_fail_with_string():
    engine = Engine(update_fn)
    pbar = ProgressBar(ncols=80)

    with pytest.raises(TypeError):
        pbar.attach(engine, "a")


def test_pbar_batch_indeces(capsys):
    engine = Engine(lambda e, b: time.sleep(0.1))

    @engine.on(Events.ITERATION_STARTED)
    def print_iter(_):
        print("iteration: ", engine.state.iteration)

    ProgressBar(persist=True, ncols=80).attach(engine)
    engine.run(list(range(4)), max_epochs=1)

    captured = capsys.readouterr()
    err = captured.err.split("\r")
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    printed_batch_indeces = set(map(lambda x: int(x.split("/")[0][-1]), err))
    expected_batch_indeces = list(range(1, 5))
    assert sorted(list(printed_batch_indeces)) == expected_batch_indeces


def test_pbar_with_metric(capsys):
    n_iters = 2
    data = list(range(n_iters))
    loss_values = iter(range(n_iters))

    def step(engine, batch):
        loss_value = next(loss_values)
        return loss_value

    trainer = Engine(step)

    RunningAverage(alpha=0.5, output_transform=lambda x: x).attach(trainer, "batchloss")

    pbar = ProgressBar(ncols=80)
    pbar.attach(trainer, metric_names=["batchloss"])

    trainer.run(data=data, max_epochs=1)

    captured = capsys.readouterr()
    err = captured.err.split("\r")
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    actual = err[-1]
    if get_tqdm_version() < Version("4.49.0"):
        expected = "Iteration: [1/2]  50%|██████     , batchloss=0.5 [00:00<00:00]"
    else:
        expected = "Iteration: [1/2]  50%|████████████████▌                , batchloss=0.5 [00:00<?]"
    assert actual == expected


def test_pbar_with_all_metric(capsys):
    n_iters = 2
    data = list(range(n_iters))
    loss_values = iter(range(n_iters))
    another_loss_values = iter(range(1, n_iters + 1))

    def step(engine, batch):
        loss_value = next(loss_values)
        another_loss_value = next(another_loss_values)
        return loss_value, another_loss_value

    trainer = Engine(step)

    RunningAverage(alpha=0.5, output_transform=lambda x: x[0]).attach(trainer, "batchloss")
    RunningAverage(alpha=0.5, output_transform=lambda x: x[1]).attach(trainer, "another batchloss")

    pbar = ProgressBar(ncols=80)
    pbar.attach(trainer, metric_names="all")

    trainer.run(data=data, max_epochs=1)

    captured = capsys.readouterr()
    err = captured.err.split("\r")
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    actual = err[-1]
    if get_tqdm_version() < Version("4.49.0"):
        expected = "Iteration: [1/2]  50%|███   , batchloss=0.5, another batchloss=1.5 [00:00<00:00]"
    else:
        expected = "Iteration: [1/2]  50%|█████     , batchloss=0.5, another batchloss=1.5 [00:00<?]"
    assert actual == expected


def test_pbar_with_state_attrs(capsys):
    n_iters = 2
    data = list(range(n_iters))
    loss_values = iter(range(n_iters))

    def step(engine, batch):
        loss_value = next(loss_values)
        return loss_value

    trainer = Engine(step)
    trainer.state.alpha = 3.899
    trainer.state.beta = torch.tensor(12.21)
    trainer.state.gamma = torch.tensor([21.0, 6.0])

    RunningAverage(alpha=0.5, output_transform=lambda x: x).attach(trainer, "batchloss")

    pbar = ProgressBar(ncols=80)
    pbar.attach(trainer, metric_names=["batchloss"], state_attributes=["alpha", "beta", "gamma"])

    trainer.run(data=data, max_epochs=1)

    captured = capsys.readouterr()
    err = captured.err.split("\r")
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    actual = err[-1]
    if get_tqdm_version() < Version("4.49.0"):
        expected = (
            "Iteration: [1/2]  50%|█████     , batchloss=0.5, alpha=3.9, beta=12.2, gamma_0=21, gamma_1=6 [00:00<00:00]"
        )
    else:
        expected = "Iteration: [1/2]  50%|▌, batchloss=0.5, alpha=3.9, beta=12.2, gamma_0=21, gamma_"
    assert actual == expected


def test_pbar_no_metric_names(capsys):
    n_epochs = 2
    loader = [1, 2]
    engine = Engine(update_fn)

    pbar = ProgressBar(ncols=80)
    pbar.attach(engine)

    engine.run(loader, max_epochs=n_epochs)

    captured = capsys.readouterr()
    err = captured.err.split("\r")
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    actual = err[-1]
    if get_tqdm_version() < Version("4.49.0"):
        expected = "Epoch [2/2]: [1/2]  50%|██████████            [00:00<00:00]"
    else:
        expected = "Epoch [2/2]: [1/2]  50%|███████████████████████                        [00:00<?]"
    assert actual == expected


def test_pbar_with_output(capsys):
    n_epochs = 2
    loader = [1, 2]
    engine = Engine(update_fn)

    pbar = ProgressBar(ncols=80)
    pbar.attach(engine, output_transform=lambda x: {"a": x})

    engine.run(loader, max_epochs=n_epochs)

    captured = capsys.readouterr()
    err = captured.err.split("\r")
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    if get_tqdm_version() < Version("4.49.0"):
        expected = "Epoch [2/2]: [1/2]  50%|█████     , a=1 [00:00<00:00]"
    else:
        expected = "Epoch [2/2]: [1/2]  50%|████████████████████▌                    , a=1 [00:00<?]"
    assert err[-1] == expected


def test_pbar_fail_with_non_callable_transform():
    engine = Engine(update_fn)
    pbar = ProgressBar(ncols=80)

    with pytest.raises(TypeError):
        pbar.attach(engine, output_transform=1)


def test_pbar_with_scalar_output(capsys):
    n_epochs = 2
    loader = [1, 2]
    engine = Engine(update_fn)

    pbar = ProgressBar(ncols=80)
    pbar.attach(engine, output_transform=lambda x: x)

    engine.run(loader, max_epochs=n_epochs)

    captured = capsys.readouterr()
    err = captured.err.split("\r")
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    if get_tqdm_version() < Version("4.49.0"):
        expected = "Epoch [2/2]: [1/2]  50%|█████     , output=1 [00:00<00:00]"
    else:
        expected = "Epoch [2/2]: [1/2]  50%|██████████████████                  , output=1 [00:00<?]"
    assert err[-1] == expected


def test_pbar_with_str_output(capsys):
    n_epochs = 2
    loader = [1, 2]
    engine = Engine(update_fn)

    pbar = ProgressBar(ncols=80)
    pbar.attach(engine, output_transform=lambda x: "red")

    engine.run(loader, max_epochs=n_epochs)

    captured = capsys.readouterr()
    err = captured.err.split("\r")
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    if get_tqdm_version() < Version("4.49.0"):
        expected = "Epoch [2/2]: [1/2]  50%|█████     , output=red [00:00<00:00]"
    else:
        expected = "Epoch [2/2]: [1/2]  50%|█████████████████                 , output=red [00:00<?]"
    assert err[-1] == expected


def test_pbar_with_tqdm_kwargs(capsys):
    n_epochs = 10
    loader = [1, 2, 3, 4, 5]
    engine = Engine(update_fn)

    pbar = ProgressBar(desc="My description: ", ncols=80)
    pbar.attach(engine, output_transform=lambda x: x)
    engine.run(loader, max_epochs=n_epochs)

    captured = capsys.readouterr()
    err = captured.err.split("\r")
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    expected = "My description:  [10/10]: [4/5]  80%|███████████████▏   , output=1 [00:00<00:00]"
    assert err[-1] == expected


def test_pbar_for_validation(capsys):
    loader = [1, 2, 3, 4, 5]
    engine = Engine(update_fn)

    pbar = ProgressBar(desc="Validation", ncols=80)
    pbar.attach(engine)
    engine.run(loader, max_epochs=1)

    captured = capsys.readouterr()
    err = captured.err.split("\r")
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    expected = "Validation: [4/5]  80%|██████████████████████████████████▍         [00:00<00:00]"
    assert err[-1] == expected


def test_pbar_output_tensor(capsys):
    def _test(out_tensor, out_msg):
        loader = [1, 2, 3, 4, 5]

        def update_fn(engine, batch):
            return out_tensor

        engine = Engine(update_fn)

        pbar = ProgressBar(desc="Output tensor", ncols=80)
        pbar.attach(engine, output_transform=lambda x: x)
        engine.run(loader, max_epochs=1)

        captured = capsys.readouterr()
        err = captured.err.split("\r")
        err = list(map(lambda x: x.strip(), err))
        err = list(filter(None, err))
        expected = f"Output tensor: [4/5]  {out_msg} [00:00<00:00]"
        assert err[-1] == expected

    _test(out_tensor=torch.tensor([5, 0]), out_msg="80%|████████████▊   , output_0=5, output_1=0")
    _test(out_tensor=torch.tensor(123), out_msg="80%|██████████████████████▍     , output=123")
    _test(out_tensor=torch.tensor(1.234), out_msg="80%|█████████████████████▌     , output=1.23")


def test_pbar_output_warning(capsys):
    loader = [1, 2, 3, 4, 5]

    def update_fn(engine, batch):
        return torch.zeros(1, 2, 3, 4)

    engine = Engine(update_fn)

    pbar = ProgressBar(desc="Output tensor", ncols=80)
    pbar.attach(engine, output_transform=lambda x: x)
    with pytest.warns(UserWarning):
        engine.run(loader, max_epochs=1)


def test_pbar_on_epochs(capsys):
    n_epochs = 10
    loader = [1, 2, 3, 4, 5]
    engine = Engine(update_fn)

    pbar = ProgressBar(ncols=80)
    pbar.attach(engine, event_name=Events.EPOCH_STARTED, closing_event_name=Events.COMPLETED)
    engine.run(loader, max_epochs=n_epochs)

    captured = capsys.readouterr()
    err = captured.err.split("\r")
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    actual = err[-1]
    expected = "Epoch: [9/10]  90%|██████████████████████████████████████████▎     [00:00<00:00]"
    assert actual == expected


def test_pbar_with_max_epochs_set_to_one(capsys):
    n_epochs = 1
    loader = [1, 2]
    engine = Engine(update_fn)

    pbar = ProgressBar(ncols=80)
    pbar.attach(engine, ["a"])

    engine.run(loader, max_epochs=n_epochs)

    captured = capsys.readouterr()
    err = captured.err.split("\r")
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    if get_tqdm_version() < Version("4.49.0"):
        expected = "Iteration: [1/2]  50%|█████     , a=1 [00:00<00:00]"
    else:
        expected = "Iteration: [1/2]  50%|█████████████████████▌                     , a=1 [00:00<?]"
    assert err[-1] == expected


def test_pbar_wrong_events_order():
    engine = Engine(update_fn)
    pbar = ProgressBar(ncols=80)

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

    with pytest.raises(ValueError, match="should not be a filtered event"):
        pbar.attach(engine, event_name=Events.ITERATION_STARTED, closing_event_name=Events.EPOCH_COMPLETED(every=10))


def test_pbar_with_nan_input():
    def update(engine, batch):
        x = batch
        return x.item()

    def create_engine():
        engine = Engine(update)
        pbar = ProgressBar(ncols=80)

        engine.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        pbar.attach(engine, event_name=Events.EPOCH_COMPLETED, closing_event_name=Events.COMPLETED)
        return engine

    data = torch.from_numpy(np.array([np.nan] * 25))
    engine = create_engine()
    engine.run(data)
    assert engine.should_terminate
    assert engine.state.iteration == 1
    assert engine.state.epoch == 1

    data = torch.from_numpy(np.array([1] * 1000 + [np.nan] * 25))
    engine = create_engine()
    engine.run(data)
    assert engine.should_terminate
    assert engine.state.iteration == 1001
    assert engine.state.epoch == 1


def test_pbar_on_callable_events(capsys):
    n_epochs = 1
    loader = list(range(100))
    engine = Engine(update_fn)

    pbar = ProgressBar(ncols=80)
    pbar.attach(engine, event_name=Events.ITERATION_STARTED(every=10), closing_event_name=Events.EPOCH_COMPLETED)
    engine.run(loader, max_epochs=n_epochs)

    captured = capsys.readouterr()
    err = captured.err.split("\r")
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    actual = err[-1]
    expected = "Iteration: [90/100]  90%|████████████████████████████████████▉     [00:00<00:00]"
    assert actual == expected


def test_tqdm_logger_epoch_length(capsys):
    loader = list(range(100))
    engine = Engine(update_fn)
    pbar = ProgressBar(persist=True, ncols=80)
    pbar.attach(engine)
    engine.run(loader, epoch_length=50)

    captured = capsys.readouterr()
    err = captured.err.split("\r")
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    actual = err[-1]
    expected = "Iteration: [50/50] 100%|██████████████████████████████████████████ [00:00<00:00]"
    assert actual == expected


def test_tqdm_logger_iter_without_epoch_length(capsys):
    size = 11

    def finite_size_data_iter(size):
        for i in range(size):
            yield i

    def train_step(trainer, batch):
        pass

    trainer = Engine(train_step)

    @trainer.on(Events.ITERATION_COMPLETED(every=size))
    def restart_iter():
        trainer.state.dataloader = finite_size_data_iter(size)

    pbar = ProgressBar(persist=True, ncols=80)
    pbar.attach(trainer)

    data_iter = finite_size_data_iter(size)
    trainer.run(data_iter, max_epochs=5)

    captured = capsys.readouterr()
    err = captured.err.split("\r")
    err = list(map(lambda x: x.strip(), err))
    err = list(filter(None, err))
    actual = err[-1]
    expected = "Epoch [5/5]: [11/11] 100%|████████████████████████████████████████ [00:00<00:00]"
    assert actual == expected
