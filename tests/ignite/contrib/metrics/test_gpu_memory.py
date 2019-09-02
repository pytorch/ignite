import sys

import torch

from ignite.engine import Engine, State
from ignite.contrib.metrics import GpuMemory

import pytest


@pytest.fixture
def no_site_packages():
    import pynvml
    import sys
    assert 'pynvml' in sys.modules
    pynvml_module = sys.modules['pynvml']
    del sys.modules['pynvml']
    prev_path = list(sys.path)
    sys.path = [p for p in sys.path if "site-packages" not in p]
    yield "no_site_packages"
    sys.path = prev_path
    sys.modules['pynvml'] = pynvml_module


@pytest.mark.skipif(sys.version[0] == "2", reason="No pynvml for python 2.7")
def test_no_pynvml_package(no_site_packages):

    with pytest.raises(RuntimeError, match="This contrib module requires pynvml to be installed."):
        GpuMemory()


@pytest.mark.skipif(sys.version[0] == "2" or torch.cuda.is_available(), reason="No pynvml for python 2.7")
def test_no_gpu():

    with pytest.raises(RuntimeError, match="This contrib module requires available GPU"):
        GpuMemory()


@pytest.mark.skipif(sys.version[0] == "2" or not (torch.cuda.is_available()),
                    reason="No pynvml for python 2.7 and no GPU")
def test_gpu_mem_consumption():

    gpu_mem = GpuMemory()

    t = torch.rand(4, 10, 100, 100)
    data = gpu_mem.compute()
    assert len(data) > 0
    assert "fb_memory_usage" in data[0]
    report = data[0]['fb_memory_usage']
    assert 'used' in report and 'total' in report
    assert report['total'] > 0.0
    assert report['used'] > t.shape[0] * t.shape[1] * t.shape[2] * t.shape[3] / 1024.0 / 1024.0

    # with Engine
    engine = Engine(lambda engine, batch: 0.0)
    engine.state = State(metrics={})

    gpu_mem.completed(engine, name='gpu mem', local_rank=0)

    assert 'gpu mem' in engine.state.metrics
    assert isinstance(engine.state.metrics['gpu mem'], str)
    assert "{}".format(int(report['used'])) in engine.state.metrics['gpu mem']
    assert "{}".format(int(report['total'])) in engine.state.metrics['gpu mem']
