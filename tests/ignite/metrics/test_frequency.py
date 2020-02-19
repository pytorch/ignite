import time

import pytest

import torch.distributed as dist

from ignite.engine import Engine, Events
from ignite.metrics import Frequency


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


def _test_frequency_with_engine(device, workers):

    artificial_time = 0.1 / workers  # seconds
    total_tokens = 1200 // workers
    batch_size = 128 // workers

    estimated_wps = batch_size * workers / artificial_time

    def update_fn(engine, batch):
        time.sleep(artificial_time)
        return {"ntokens": len(batch)}

    engine = Engine(update_fn)
    wps_metric = Frequency(output_transform=lambda x: x["ntokens"], device=device)
    wps_metric.attach(engine, "wps")

    @engine.on(Events.ITERATION_COMPLETED)
    def assert_wps(e):
        wps = e.state.metrics["wps"]
        assert estimated_wps * 0.85 < wps < estimated_wps, "{}: {} < {} < {}".format(
            e.state.iteration, estimated_wps * 0.85, wps, estimated_wps
        )

    data = [[i] * batch_size for i in range(0, total_tokens, batch_size)]
    engine.run(data, max_epochs=1)


def test_frequency_with_engine():
    device = "cpu"
    _test_frequency_with_engine(device, workers=1)


@pytest.mark.distributed
def test_frequency_with_engine_distributed(distributed_context_single_node_gloo):
    device = "cpu"
    _test_frequency_with_engine(device, workers=dist.get_world_size())
