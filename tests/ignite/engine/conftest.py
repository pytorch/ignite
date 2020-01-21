import pytest


class IterationCounter(object):
    def __init__(self, start_value=1):
        self.current_iteration_count = start_value

    def __call__(self, engine):
        assert engine.state.iteration == self.current_iteration_count
        self.current_iteration_count += 1


class EpochCounter(object):
    def __init__(self, start_value):
        self.current_epoch_count = start_value

    def __call__(self, engine):
        assert engine.state.epoch == self.current_epoch_count
        self.current_epoch_count += 1


@pytest.fixture()
def counter_factory():

    def create(name, start_value=1):
        if name == "epoch":
            return EpochCounter(start_value)
        elif name == "iter":
            return IterationCounter(start_value)
        else:
            raise RuntimeError()

    return create
