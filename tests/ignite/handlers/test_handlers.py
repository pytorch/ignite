from unittest.mock import MagicMock

from ignite.engine import Engine, Events
from ignite.handlers import global_step_from_engine


def test_global_step_from_engine():
    iteration = 12
    epoch = 23

    trainer = Engine(lambda e, b: None)
    trainer.state.iteration = iteration
    trainer.state.epoch = epoch

    gst = global_step_from_engine(trainer)
    assert gst(MagicMock(), Events.EPOCH_COMPLETED) == epoch

    gst = global_step_from_engine(trainer, custom_event_name=Events.ITERATION_COMPLETED)
    assert gst(MagicMock(), Events.EPOCH_COMPLETED) == iteration
