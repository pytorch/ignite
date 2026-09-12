import time
from types import SimpleNamespace

import pytest

from ignite.engine import Engine, Events
from ignite.handlers import TimeLimit


@pytest.mark.parametrize("wall_elapsed, elapsed, terminated", [(1000, 2, False), (-1000, 11, True)])
def test_time_limit_ignores_wall_clock_adjustments(monkeypatch, wall_elapsed, elapsed, terminated):
    import ignite.handlers.time_limit as module

    clocks = {"wall": 10000, "elapsed": 20}
    monkeypatch.setattr(
        module, "time", SimpleNamespace(time=lambda: clocks["wall"], monotonic=lambda: clocks["elapsed"])
    )
    handler = TimeLimit(limit_sec=10)
    trainer = Engine(lambda engine, batch: batch)
    clocks["wall"] += wall_elapsed
    clocks["elapsed"] += elapsed

    handler(trainer)

    assert trainer.should_terminate is terminated


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
