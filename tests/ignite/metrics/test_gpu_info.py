from unittest.mock import Mock, patch

import pytest
import torch

from ignite.engine import Engine, State

from ignite.metrics import GpuInfo


def test_no_pynvml_package():
    with patch.dict("sys.modules", {"pynvml.smi": None}):
        with pytest.raises(ModuleNotFoundError, match="This contrib module requires pynvml to be installed."):
            GpuInfo()


@pytest.mark.skipif(torch.cuda.is_available(), reason="Skip if GPU")
def test_no_gpu():
    with pytest.raises(RuntimeError, match="This contrib module requires available GPU"):
        GpuInfo()


def _test_gpu_info(device="cpu"):
    gpu_info = GpuInfo()

    # increase code cov
    gpu_info.reset()
    gpu_info.update(None)

    t = torch.rand(4, 10, 100, 100).to(device)
    data = gpu_info.compute()
    assert len(data) > 0
    assert "fb_memory_usage" in data[0]
    mem_report = data[0]["fb_memory_usage"]
    assert "used" in mem_report and "total" in mem_report
    assert mem_report["total"] > 0.0
    assert mem_report["used"] > t.shape[0] * t.shape[1] * t.shape[2] * t.shape[3] / 1024.0 / 1024.0

    assert "utilization" in data[0]
    util_report = data[0]["utilization"]
    assert "gpu_util" in util_report

    # with Engine
    engine = Engine(lambda engine, batch: 0.0)
    engine.state = State(metrics={})

    gpu_info.completed(engine, name="gpu")

    assert "gpu:0 mem(%)" in engine.state.metrics

    assert isinstance(engine.state.metrics["gpu:0 mem(%)"], int)
    assert int(mem_report["used"] * 100.0 / mem_report["total"]) == engine.state.metrics["gpu:0 mem(%)"]

    if util_report["gpu_util"] != "N/A":
        assert "gpu:0 util(%)" in engine.state.metrics
        assert isinstance(engine.state.metrics["gpu:0 util(%)"], int)
        assert int(util_report["gpu_util"]) == engine.state.metrics["gpu:0 util(%)"]
    else:
        assert "gpu:0 util(%)" not in engine.state.metrics


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip if no GPU")
def test_gpu_info_on_cuda():
    _test_gpu_info(device="cuda")


query_resp = None


@pytest.fixture
def mock_pynvml_module():
    with patch.dict(
        "sys.modules",
        {
            "pynvml": Mock(name="pynvml"),
            "pynvml.smi": Mock(name="pynvml.smi"),
            "pynvml.smi.nvidia_smi": Mock(name="pynvml.smi.nvidia_smi"),
        },
    ):
        import pynvml
        from pynvml.smi import nvidia_smi

        def query(*args, **kwargs):
            return query_resp

        def getInstance():
            nvsmi = Mock()
            nvsmi.DeviceQuery = Mock(side_effect=query)
            return nvsmi

        nvidia_smi.getInstance = Mock(side_effect=getInstance)
        yield pynvml


@pytest.fixture
def mock_gpu_is_available():
    with patch("ignite.metrics.gpu_info.torch.cuda") as mock_cuda:
        mock_cuda.is_available.return_value = True
        yield mock_cuda


@pytest.mark.skipif(torch.cuda.is_available(), reason="No need to mock if has GPU")
def test_gpu_info_mock(mock_pynvml_module, mock_gpu_is_available):
    global query_resp

    query_resp = {"gpu": [{"fb_memory_usage": {"used": 100.0, "total": 11000.0}, "utilization": {"gpu_util": 50.0}}]}

    assert torch.cuda.is_available()
    _test_gpu_info()

    # Tests https://github.com/pytorch/ignite/issues/1040
    query_resp = {"gpu": [{"fb_memory_usage": {"used": 100.0, "total": 11000.0}, "utilization": {"gpu_util": "N/A"}}]}
    _test_gpu_info()

    def _test_with_custom_query(resp, warn_msg, check_compute=False):
        from pynvml.smi import nvidia_smi

        def query(*args, **kwargs):
            return resp

        def getInstance():
            nvsmi = Mock()
            nvsmi.DeviceQuery = Mock(side_effect=query)
            return nvsmi

        nvidia_smi.getInstance = Mock(side_effect=getInstance)
        gpu_info = GpuInfo()
        if check_compute:
            with pytest.warns(UserWarning, match=warn_msg):
                gpu_info.compute()

        # with Engine
        engine = Engine(lambda engine, batch: 0.0)
        engine.state = State(metrics={})

        with pytest.warns(UserWarning, match=warn_msg):
            gpu_info.completed(engine, name="gpu info")

    # No GPU info
    _test_with_custom_query(resp={}, warn_msg=r"No GPU information available", check_compute=True)

    # No GPU memory info
    _test_with_custom_query(resp={"gpu": [{"utilization": {}}]}, warn_msg=r"No GPU memory usage information available")

    # No GPU utilization info
    _test_with_custom_query(
        resp={"gpu": [{"fb_memory_usage": {}}]}, warn_msg=r"No GPU utilization information available"
    )
