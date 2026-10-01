import time

import pytest

from ignite.engine import Engine, Events
from ignite.handlers import TimeLimit


def test_arg_validation():
    with pytest.raises(ValueError, match=r"Argument limit_sec should be a positive integer."):
        TimeLimit(limit_sec=-5)

    with pytest.raises(TypeError, match=r"Argument limit_sec should be an integer."):
        TimeLimit(limit_sec="abc")


def _train_func(engine, batch):
    time.sleep(1)


@pytest.mark.parametrize("n_iters, limit", [(20, 10), (5, 10)])
def test_terminate_on_time_limit(n_iters, limit):
    started = time.time()
    trainer = Engine(_train_func)

    @trainer.on(Events.TERMINATE)
    def _():
        trainer.state.is_terminated = True

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TimeLimit(limit))
    trainer.state.is_terminated = False

    trainer.run(range(n_iters))
    elapsed = round(time.time() - started)
    assert elapsed <= limit + 1
    assert trainer.state.is_terminated == (n_iters > limit)
