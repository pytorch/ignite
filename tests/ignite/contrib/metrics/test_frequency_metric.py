import time

import pytest

from ignite.engine import Engine
from ignite.contrib.metrics import Frequency


def test_nondistributed_average():
    artificial_time = 2  # seconds
    num_tokens = 100
    average_upper_bound = num_tokens / artificial_time
    average_lower_bound = average_upper_bound * 0.9
    freq_metric = Frequency()
    freq_metric.reset()
    freq_metric.update(num_tokens)
    time.sleep(artificial_time)
    average = freq_metric.compute()
    assert average_lower_bound < average < average_upper_bound


def _test_frequency_with_engine(device):
    artificial_time = 2  # seconds
    batch_size = 4
    n_tokens = 10000
    average_upper_bound = n_tokens / artificial_time
    average_lower_bound = average_upper_bound * 0.9

    def update_fn(engine, batch):
        time.sleep(artificial_time)
        return {"ntokens": len(batch)}

    engine = Engine(update_fn)
    wps_metric = Frequency(output_transform=lambda x: x["ntokens"], device=device)
    wps_metric.attach(engine, 'wps')
    data = [list(range(n_tokens))] * batch_size
    wps = engine.run(data, max_epochs=1).metrics['wps']
    assert average_lower_bound < wps < average_upper_bound


def test_frequency_with_engine_nondistributed():
    device = "cpu"
    _test_frequency_with_engine(device)


@pytest.mark.distributed
def test_frequency_with_engine_distributed(distributed_context_single_node_gloo):
    device = "cpu"
    _test_frequency_with_engine(device)
