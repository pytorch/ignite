import time

import torch

from ignite.engine import Engine
from ignite.contrib.metrics import FrequencyMetric


def test_nondistributed_average():

    artificial_time = 2 # seconds
    num_tokens = 100
 
    freq_metric = FrequencyMetric()

    freq_metric.reset()
    freq_metric.update(num_tokens)
    time.sleep(artificial_time)
    average = freq_metric.compute()
   
    average_upper_bound = num_tokens / artificial_time
    average_lower_bound = average_upper_bound * 0.9 

    assert average_lower_bound < average < average_upper_bound

def test_frequency_with_engine():
 
    artificial_time = 2 # seconds
    size = 100
    batch_size = 10

    def update_fn(engine, batch):
        print(batch)
        time.sleep(artificial_time)
        return { "ntokens": size }

    engine = Engine(update_fn)

    wps_metric = FrequencyMetric(output_transform=lambda x: x["ntokens"])
    wps_metric.attach(engine, 'wps')
 
    data = list(range(size // batch_size))
    
    wps = engine.run(data, max_epochs=1).metrics['wps']
    
    average_upper_bound = num_tokens / artificial_time
    average_lower_bound = average_upper_bound * 0.9
    
    assert average_lower_bound < wps < average_upper_bound
