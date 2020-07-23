import os
import sys
import time

import pytest
import torch

import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.metrics import Frequency

if sys.platform.startswith("darwin"):
    pytest.skip("Skip if on MacOS", allow_module_level=True)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Skip on Windows")
def test_nondistributed_average():
    artificial_time = 1  # seconds
    num_tokens = 100
    average_upper_bound = num_tokens / artificial_time
    average_lower_bound = average_upper_bound * 0.9
    freq_metric = Frequency()
    freq_metric.reset()
    time.sleep(artificial_time)
    freq_metric.update(num_tokens)
    average = freq_metric.compute()
    assert average_lower_bound < average < average_upper_bound


def _test_frequency_with_engine(device, workers=None, lower_bound_factor=0.8, every=1):

    if workers is None:
        workers = idist.get_world_size()

    artificial_time = 1.0 / workers  # seconds
    total_tokens = 400 // workers
    batch_size = 128 // workers

    estimated_wps = batch_size * workers / artificial_time

    def update_fn(engine, batch):
        time.sleep(artificial_time)
        return {"ntokens": len(batch)}

    engine = Engine(update_fn)
    wps_metric = Frequency(output_transform=lambda x: x["ntokens"], device=device)
    event = Events.ITERATION_COMPLETED(every=every)
    wps_metric.attach(engine, "wps", event_name=event)

    @engine.on(event)
    def assert_wps(e):
        wps = e.state.metrics["wps"]
        assert estimated_wps * lower_bound_factor < wps <= estimated_wps, "{}: {} < {} < {}".format(
            e.state.iteration, estimated_wps * lower_bound_factor, wps, estimated_wps
        )

    data = [[i] * batch_size for i in range(0, total_tokens, batch_size)]
    engine.run(data, max_epochs=1)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="Skip on MacOS")
@pytest.mark.skipif(sys.platform.startswith("win"), reason="Skip on Windows")
def test_frequency_with_engine():
    device = "cpu"
    _test_frequency_with_engine(device, workers=1)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_frequency_with_engine_distributed(distributed_context_single_node_gloo):
    device = "cpu"
    _test_frequency_with_engine(device, workers=idist.get_world_size())


def test_frequency_with_engine_with_every():
    device = "cpu"
    _test_frequency_with_engine(device, workers=1, every=1)
    _test_frequency_with_engine(device, workers=1, every=10)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_native_dist_support, reason="Skip if no native dist support")
def test_frequency_with_engine_distributed_with_every(distributed_context_single_node_gloo):
    device = "cpu"
    _test_frequency_with_engine(device, workers=idist.get_world_size(), every=1)
    _test_frequency_with_engine(device, workers=idist.get_world_size(), every=10)


@pytest.mark.distributed
@pytest.mark.skipif(not idist.has_hvd_support, reason="Skip if no Horovod dist support")
@pytest.mark.skipif("WORLD_SIZE" in os.environ, reason="Skip if launched as multiproc")
def test_distrib_hvd(gloo_hvd_executor):

    device = "cpu" if not torch.cuda.is_available() else "cuda"
    nproc = 4 if not torch.cuda.is_available() else torch.cuda.device_count()

    gloo_hvd_executor(_test_frequency_with_engine, (device, None, 0.8, 1), np=nproc, do_init=True)
    gloo_hvd_executor(_test_frequency_with_engine, (device, None, 0.8, 10), np=nproc, do_init=True)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" in os.environ, reason="Skip if NUM_TPU_WORKERS is in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_single_device_xla():
    device = idist.device()
    _test_frequency_with_engine(device, workers=idist.get_world_size(), every=10)


def _test_distrib_xla_nprocs(index):
    device = idist.device()
    _test_frequency_with_engine(device, workers=idist.get_world_size(), every=10)


@pytest.mark.tpu
@pytest.mark.skipif("NUM_TPU_WORKERS" not in os.environ, reason="Skip if no NUM_TPU_WORKERS in env vars")
@pytest.mark.skipif(not idist.has_xla_support, reason="Skip if no PyTorch XLA package")
def test_distrib_xla_nprocs(xmp_executor):
    n = int(os.environ["NUM_TPU_WORKERS"])
    xmp_executor(_test_distrib_xla_nprocs, args=(), nprocs=n)
