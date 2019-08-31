import torch

from ignite.contrib.metrics import GpuMemory

import pytest


@pytest.fixture
def no_site_packages():
    try:
        import pynvml
    except ImportError:
        yield "no_site_packages"
        return
    import sys
    assert 'pynvml' in sys.modules
    pynvml_module = sys.modules['pynvml']
    del sys.modules['pynvml']
    prev_path = list(sys.path)
    sys.path = [p for p in sys.path if "site-packages" not in p]
    yield "no_site_packages"
    sys.path = prev_path
    sys.modules['pynvml'] = pynvml_module


def test_no_pynvml_package(no_site_packages):

    with pytest.raises(RuntimeError, match="This contrib module requires pynvml to be installed."):
        GpuMemory()


@pytest.mark.skipif(torch.cuda.is_available(), reason="Skip if has GPU")
def test_no_gpu():

    with pytest.raises(RuntimeError, match="This contrib module requires available GPU"):
        GpuMemory()


@pytest.mark.skipif(torch.cuda.is_available(), reason="Skip if has GPU")
def test_gpu_mem_consumption():

    gpu_mem = GpuMemory()

    data = gpu_mem.compute()
    assert len(data) > 0
