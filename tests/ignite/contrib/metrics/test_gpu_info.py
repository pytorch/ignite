import sys

import torch

from ignite.engine import Engine, State
from ignite.contrib.metrics import GpuInfo

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
        GpuInfo()


@pytest.mark.skipif(sys.version[0] == "2" or torch.cuda.is_available(), reason="No pynvml for python 2.7")
def test_no_gpu():

    with pytest.raises(RuntimeError, match="This contrib module requires available GPU"):
        GpuInfo()


@pytest.mark.skipif(sys.version[0] == "2" or not (torch.cuda.is_available()),
                    reason="No pynvml for python 2.7 and no GPU")
def test_gpu_mem_consumption():

    gpu_info = GpuInfo()

    t = torch.rand(4, 10, 100, 100)
    data = gpu_info.compute()
    assert len(data) > 0
    assert "fb_memory_usage" in data[0]
    mem_report = data[0]['fb_memory_usage']
    assert 'used' in mem_report and 'total' in mem_report
    assert mem_report['total'] > 0.0
    assert mem_report['used'] > t.shape[0] * t.shape[1] * t.shape[2] * t.shape[3] / 1024.0 / 1024.0

    assert "utilization" in data[0]
    util_report = data[0]['utilization']
    assert 'gpu_util' in util_report

    # with Engine
    engine = Engine(lambda engine, batch: 0.0)
    engine.state = State(metrics={})

    gpu_info.completed(engine, name='gpu info')

    assert 'gpu info:0 memory' in engine.state.metrics
    assert 'gpu info:0 util' in engine.state.metrics

    assert isinstance(engine.state.metrics['gpu info:0 memory'], str)
    assert "{}".format(int(mem_report['used'])) in engine.state.metrics['gpu info:0 memory']
    assert "{}".format(int(mem_report['total'])) in engine.state.metrics['gpu info:0 memory']

    assert isinstance(engine.state.metrics['gpu info:0 util'], str)
    assert "{}".format(int(util_report['gpu_util'])) in engine.state.metrics['gpu info:0 util']
