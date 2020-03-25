import torch
from torch import nn
from ignite.engine import create_supervised_evaluator
from ignite.contrib.handlers import ThroughputBenchmarkWrapper
import pytest

NUM_ITERS = 1
NUM_WARMUP_ITERS = 0


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc1(x)


@pytest.fixture
def model():
    model = DummyModel()
    yield model


@pytest.fixture
def throughput_benchmark_wrapper(model):
    yield ThroughputBenchmarkWrapper(model, num_iters=NUM_ITERS, num_warmup_iters=NUM_WARMUP_ITERS)


@pytest.fixture
def dataloader():
    yield torch.rand(100, 2, 1)


@pytest.fixture
def dummy_cpu_engine(model):
    engine = create_supervised_evaluator(model, device="cpu")
    yield engine


def test_throughput_inputs(model):
    with pytest.raises(TypeError):
        ThroughputBenchmarkWrapper(model, num_calling_threads="1")

    with pytest.raises(TypeError):
        ThroughputBenchmarkWrapper(model, num_warmup_iters="1")

    with pytest.raises(TypeError):
        ThroughputBenchmarkWrapper(model, num_iters="1")

    with pytest.raises(ValueError):
        ThroughputBenchmarkWrapper(model, num_calling_threads=0)

    with pytest.raises(ValueError):
        ThroughputBenchmarkWrapper(model, num_warmup_iters=-1)

    with pytest.raises(ValueError):
        ThroughputBenchmarkWrapper(model, num_iters=0)


def test_attach_inputs(model, throughput_benchmark_wrapper, dummy_cpu_engine):
    with pytest.raises(TypeError):
        with throughput_benchmark_wrapper.attach(dummy_cpu_engine, max_batches="10") as t:
            pass
    with pytest.raises(TypeError):
        with throughput_benchmark_wrapper.attach(dummy_cpu_engine, input_transform=2):
            pass
    with pytest.raises(ValueError):
        with throughput_benchmark_wrapper.attach(dummy_cpu_engine, max_batches=0) as t:
            pass


def test_no_run(model, throughput_benchmark_wrapper, dummy_cpu_engine):
    with throughput_benchmark_wrapper.attach(dummy_cpu_engine) as t:
        pass
    with pytest.raises(RuntimeError):
        throughput_benchmark_wrapper.stats


def test_detach(model, throughput_benchmark_wrapper, dummy_cpu_engine, dataloader):
    with throughput_benchmark_wrapper.attach(dummy_cpu_engine) as t:
        t.run(dataloader)

    for event in dummy_cpu_engine._event_handlers:
        assert len(dummy_cpu_engine._event_handlers[event]) == 0


def test_working_cpu_model(model, throughput_benchmark_wrapper, dataloader, dummy_cpu_engine):
    with throughput_benchmark_wrapper.attach(dummy_cpu_engine) as bench_wrapper:
        bench_wrapper.run(dataloader)

    assert throughput_benchmark_wrapper.stats.num_iters == NUM_ITERS
    assert throughput_benchmark_wrapper.stats.latency_avg_ms < 100
