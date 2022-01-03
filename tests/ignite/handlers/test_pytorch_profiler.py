import glob
import os

import pytest
import torch

import ignite.distributed as idist
from ignite.engine import Engine
from ignite.handlers import PyTorchProfiler


def update_fn(engine, batch):
    a = torch.empty((2, 3), dtype=torch.int32)
    b = torch.empty((3, 3), dtype=torch.int32)

    return a + torch.mm(a, b)


def get_engine():
    dummy_trainer = Engine(update_fn)
    return dummy_trainer


def test_get_results(tmp_path):
    trainer = get_engine()
    pt_profiler = PyTorchProfiler(on_trace_ready="tensorboard", output_path=tmp_path)
    pt_profiler.attach(trainer)
    trainer.run(range(10), max_epochs=1)

    with pytest.raises(ValueError, match=r" The sort_key cpu_times is not accepted. Please choose a sort key from"):
        pt_profiler.get_results(sort_key="cpu_times")


def test_write_results(tmp_path):
    n = 5

    trainer = get_engine()
    pt_profiler = PyTorchProfiler(on_trace_ready="tensorboard", output_path=tmp_path)
    pt_profiler.attach(trainer)
    trainer.run(range(10), max_epochs=1)
    pt_profiler.write_results(n=n)

    fp = glob.glob(os.path.join(tmp_path, f"{idist.backend()}_*"))[0 - 1]
    assert os.path.isfile(fp)

    file_length = 0
    with open(fp, "r") as fp:
        for _ in fp:
            file_length += 1

    assert file_length == n + 5
