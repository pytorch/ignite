import pytest

from ignite.engine import Engine
from ignite.contrib.handlers.custom_epoch import CustomEpochLength, CustomEpochEvents


def test_bad_input():

    with pytest.raises(ValueError):
        CustomEpochLength(n_iterations="a")

    with pytest.raises(ValueError):
        CustomEpochLength(n_iterations=0)

    with pytest.raises(ValueError):
        CustomEpochLength(n_iterations=10.0)


def test_integration():

    loss_values = list(range(500, 0, -1))
    loss_iter = iter(loss_values)

    def update(*args, **kwargs):
        return next(loss_iter)

    engine = Engine(update)

    n_iterations = 3
    CustomEpochLength(n_iterations=n_iterations).attach(engine)
    data = list(range(16))

    custom_epoch = [1]

    @engine.on(CustomEpochEvents.CUSTOM_EPOCH_STARTED)
    def on_my_epoch_started(engine):
        # print("Custom epoch {} started at iteration {}".format(engine.state.custom_epoch, engine.state.iteration))
        assert (engine.state.iteration - 1) % n_iterations == 0
        assert engine.state.custom_epoch == custom_epoch[0]

    @engine.on(CustomEpochEvents.CUSTOM_EPOCH_COMPLETED)
    def on_my_epoch_ended(engine):
        # print("Custom epoch {} ended at iteration {}".format(engine.state.custom_epoch, engine.state.iteration))
        assert engine.state.iteration % n_iterations == 0
        assert engine.state.custom_epoch == custom_epoch[0]
        custom_epoch[0] += 1

    engine.run(data, max_epochs=2)

    assert custom_epoch[0] == 11
