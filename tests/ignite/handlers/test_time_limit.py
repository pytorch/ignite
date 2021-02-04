import time

import pytest
import torch

from ignite.engine import Engine, Events
from ignite.handlers import TimeLimit


def test_arg_validation():

    with pytest.raises(ValueError, match=r"Argument limit_sec should be a positive number."):
        TimeLimit(limit_sec=-5)


def test_terminate_on_time_limit():

    sleep_time = 1

    def _train_func(engine, batch):
        time.sleep(sleep_time)

    trainer = Engine(_train_func)

    n_iter = 3
    limit = 2

    low_limit = TimeLimit(limit)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, low_limit)

    trainer.run(range(n_iter))
    assert trainer.state.iteration == limit

    n_iter = 2
    limit = 3
    high_limit = TimeLimit(limit)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, high_limit)

    trainer.run(range(n_iter))
    assert trainer.state.iteration < limit
