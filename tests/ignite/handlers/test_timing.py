import sys
import time

import pytest

from ignite.engine import Engine, Events
from ignite.handlers import Timer

if sys.platform.startswith("darwin"):
    pytest.skip("Skip if on MacOS", allow_module_level=True)


def test_timer():
    sleep_t = 0.2
    n_iter = 3

    def _train_func(engine, batch):
        time.sleep(sleep_t)

    def _test_func(engine, batch):
        time.sleep(sleep_t)

    trainer = Engine(_train_func)
    tester = Engine(_test_func)

    t_total = Timer()
    t_batch = Timer(average=True)
    t_train = Timer()

    t_total.attach(trainer)
    t_batch.attach(
        trainer, pause=Events.ITERATION_COMPLETED, resume=Events.ITERATION_STARTED, step=Events.ITERATION_COMPLETED
    )
    t_train.attach(trainer, pause=Events.EPOCH_COMPLETED, resume=Events.EPOCH_STARTED)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(trainer):
        tester.run(range(n_iter))

    # Run "training"
    trainer.run(range(n_iter))

    def _equal(lhs, rhs):
        return round(lhs, 1) == round(rhs, 1)

    assert _equal(t_total.value(), (2 * n_iter * sleep_t))
    assert _equal(t_batch.value(), (sleep_t))
    assert _equal(t_train.value(), (n_iter * sleep_t))

    t_total.reset()
    assert _equal(t_total.value(), 0.0)
