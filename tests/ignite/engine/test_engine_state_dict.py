from collections.abc import Mapping

import pytest
import torch

from ignite.engine import Engine, Events, State
from tests.ignite.engine import BatchChecker, EpochCounter, IterationCounter


def test_state_dict():
    engine = Engine(lambda e, b: 1)
    sd = engine.state_dict()
    assert isinstance(sd, Mapping) and len(sd) == 3
    assert "iteration" in sd and sd["iteration"] == 0
    assert "max_epochs" in sd and sd["max_epochs"] is None
    assert "epoch_length" in sd and sd["epoch_length"] is None

    def _test(state):
        engine.state = state
        sd = engine.state_dict()
        assert isinstance(sd, Mapping) and len(sd) == len(engine._state_dict_all_req_keys) + 1
        assert sd["iteration"] == engine.state.iteration
        assert sd["epoch_length"] == engine.state.epoch_length
        assert sd["max_epochs"] == engine.state.max_epochs

    _test(State(iteration=500, epoch_length=1000, max_epochs=100))
    _test(State(epoch=5, epoch_length=1000, max_epochs=100))


def test_state_dict_with_user_keys():
    engine = Engine(lambda e, b: 1)
    engine.state_dict_user_keys.append("alpha")
    engine.state_dict_user_keys.append("beta")

    def _test(state):
        engine.state = state
        sd = engine.state_dict()
        assert isinstance(sd, Mapping) and len(sd) == len(engine._state_dict_all_req_keys) + 1 + len(
            engine.state_dict_user_keys
        )
        assert sd["iteration"] == engine.state.iteration
        assert sd["epoch_length"] == engine.state.epoch_length
        assert sd["max_epochs"] == engine.state.max_epochs
        assert sd["alpha"] == engine.state.alpha
        assert sd["beta"] == engine.state.beta

    _test(State(iteration=500, epoch_length=1000, max_epochs=100, alpha=0.01, beta="Good"))


def test_state_dict_integration():
    engine = Engine(lambda e, b: 1)
    data = range(100)
    engine.run(data, max_epochs=10)
    sd = engine.state_dict()
    assert isinstance(sd, Mapping) and len(sd) == len(engine._state_dict_all_req_keys) + 1
    assert sd["iteration"] == engine.state.iteration == 10 * 100
    assert sd["epoch_length"] == engine.state.epoch_length == 100
    assert sd["max_epochs"] == engine.state.max_epochs == 10


def test_load_state_dict_asserts():
    engine = Engine(lambda e, b: 1)

    with pytest.raises(TypeError, match=r"Argument state_dict should be a dictionary"):
        engine.load_state_dict("123")

    with pytest.raises(ValueError, match=r"is absent in provided state_dict"):
        engine.load_state_dict({})

    with pytest.raises(ValueError, match=r"state_dict should contain only one of"):
        engine.load_state_dict({"max_epochs": 100, "epoch_length": 120})

    with pytest.raises(ValueError, match=r"state_dict should contain only one of"):
        engine.load_state_dict({"max_epochs": 100, "epoch_length": 120, "iteration": 12, "epoch": 123})

    engine = Engine(lambda e, b: 1)
    engine.state_dict_user_keys.append("alpha")
    with pytest.raises(ValueError, match=r"Required user state attribute"):
        engine.load_state_dict({"max_epochs": 100, "epoch_length": 120, "iteration": 12})

    engine = Engine(lambda e, b: 1)
    with pytest.raises(ValueError, match=r"If epoch is provided in the state dict, epoch_length should not be None"):
        engine.load_state_dict({"max_epochs": 100, "epoch": 2, "epoch_length": None})


def test_load_state_dict():
    engine = Engine(lambda e, b: 1)

    def _test(sd):
        engine.load_state_dict(sd)
        if "iteration" in sd:
            assert sd["iteration"] == engine.state.iteration
        elif "epoch" in sd:
            assert sd["epoch"] == engine.state.epoch
        assert sd["epoch_length"] == engine.state.epoch_length
        assert sd["max_epochs"] == engine.state.max_epochs

    _test({"max_epochs": 100, "epoch_length": 120, "iteration": 123})
    _test({"max_epochs": 100, "epoch_length": 120, "epoch": 5})


def test_load_state_dict_with_user_keys():
    engine = Engine(lambda e, b: 1)
    engine.state_dict_user_keys.append("alpha")
    engine.state_dict_user_keys.append("beta")

    def _test(sd):
        engine.load_state_dict(sd)
        if "iteration" in sd:
            assert sd["iteration"] == engine.state.iteration
        elif "epoch" in sd:
            assert sd["epoch"] == engine.state.epoch
        assert sd["epoch_length"] == engine.state.epoch_length
        assert sd["max_epochs"] == engine.state.max_epochs
        assert sd["alpha"] == engine.state.alpha
        assert sd["beta"] == engine.state.beta

    _test({"max_epochs": 100, "epoch_length": 120, "iteration": 123, "alpha": 0.1, "beta": "abc"})


def test_load_state_dict_integration():
    engine = Engine(lambda e, b: 1)

    state_dict = {"max_epochs": 100, "epoch_length": 120, "epoch": 5}

    engine.load_state_dict(state_dict)
    engine.add_event_handler(Events.ITERATION_COMPLETED, IterationCounter(5 * 120 + 1))
    engine.add_event_handler(Events.EPOCH_COMPLETED, EpochCounter(6))
    data = range(120)
    engine.run(data)


def test_load_state_dict_with_params_overriding_integration():
    state_dict = {"max_epochs": 100, "epoch_length": 120, "epoch": 5}
    data = range(120)

    # Override max_epochs
    new_max_epochs = 10
    engine = Engine(lambda e, b: 1)
    engine.load_state_dict(state_dict)
    state = engine.run(data, max_epochs=new_max_epochs)
    assert state.max_epochs == new_max_epochs
    assert state.iteration == state_dict["epoch_length"] * new_max_epochs
    assert state.epoch == new_max_epochs

    with pytest.raises(ValueError, match=r"Argument max_epochs should be greater than or equal to the start epoch"):
        engine.load_state_dict(state_dict)
        engine.run(data, max_epochs=3)

    # Override epoch_length
    with pytest.raises(ValueError, match=r"Argument epoch_length should be same as in the state"):
        engine.load_state_dict(state_dict)
        engine.run(data, epoch_length=90)


def test_empty_state_dict_load_state_dict():
    engine = Engine(lambda e, b: 1)
    sd = engine.state_dict()
    engine.load_state_dict(sd)


def test_continue_training():
    # Tests issue : https://github.com/pytorch/ignite/issues/993
    max_epochs = 2
    data = range(10)
    engine = Engine(lambda e, b: 1)
    state = engine.run(data, max_epochs=max_epochs)
    assert state.max_epochs == max_epochs
    assert state.iteration == len(data) * max_epochs
    assert state.epoch == max_epochs

    @engine.on(Events.STARTED)
    def assert_continue_training():
        assert engine.state.epoch == max_epochs

    state = engine.run(data, max_epochs=max_epochs * 2)
    assert state.max_epochs == max_epochs * 2
    assert state.iteration == len(data) * max_epochs * 2
    assert state.epoch == max_epochs * 2


def test_state_dict_with_user_keys_integration(dirname):
    engine = Engine(lambda e, b: 1)
    engine.state_dict_user_keys.append("alpha")

    @engine.on(Events.STARTED)
    def init_user_values(_):
        engine.state.alpha = 0.1

    fp = dirname / "engine.pt"

    @engine.on(Events.COMPLETED)
    def save_engine(_):
        state_dict = engine.state_dict()
        assert "alpha" in state_dict
        torch.save(state_dict, fp)

    engine.run([0, 1])

    assert fp.exists()
    state_dict = torch.load(fp)
    assert "alpha" in state_dict and state_dict["alpha"] == 0.1


def test_epoch_length():
    def _test(data, max_epochs, num_iters):
        batch_checker = BatchChecker(data)

        def update_fn(_, batch):
            assert batch_checker.check(batch), f"{batch_checker.counter}: {batch_checker.true_batch} vs {batch}"

        engine = Engine(update_fn)
        engine.run(data, max_epochs=max_epochs, epoch_length=num_iters)
        if num_iters is None:
            num_iters = len(data)
        assert engine.state.iteration == num_iters * max_epochs
        assert engine.state.epoch == max_epochs

    def _test_as_iter(data, max_epochs, num_iters):
        batch_checker = BatchChecker(data)

        def update_fn(_, batch):
            assert batch_checker.check(batch), f"{batch_checker.counter}: {batch_checker.true_batch} vs {batch}"

        engine = Engine(update_fn)
        engine.run(iter(data), max_epochs=max_epochs, epoch_length=num_iters)
        if num_iters is None:
            num_iters = len(data)
        assert engine.state.iteration == num_iters * max_epochs
        assert engine.state.epoch == max_epochs

    max_epochs = 10
    num_iters = 21
    data = torch.randint(0, 1000, size=(num_iters,))
    _test(data, max_epochs, num_iters=None)
    _test(data, max_epochs, num_iters)
    _test(data, max_epochs, num_iters // 2)
    _test(data, max_epochs, num_iters * 2)

    _test_as_iter(data, 1, num_iters)
    _test_as_iter(data, 2, num_iters // 2)


def test_state_custom_attrs_init():
    def _test(with_load_state_dict=False):
        engine = Engine(lambda e, b: None)
        engine.state.alpha = 0.0
        engine.state.beta = 1.0

        if with_load_state_dict:
            engine.load_state_dict({"iteration": 3, "max_epochs": 5, "epoch_length": 5})

        @engine.on(Events.STARTED | Events.EPOCH_STARTED | Events.EPOCH_COMPLETED | Events.COMPLETED)
        def check_custom_attr():
            assert hasattr(engine.state, "alpha") and engine.state.alpha == 0.0
            assert hasattr(engine.state, "beta") and engine.state.beta == 1.0

        engine.run([0, 1, 2, 3, 4], max_epochs=5)

    _test()
    _test(with_load_state_dict=True)


def test_restart_training():
    data = range(10)
    engine = Engine(lambda e, b: 1)
    state = engine.run(data, max_epochs=5)
    with pytest.raises(
        ValueError,
        match=r"Argument max_epochs should be greater than or equal to the start epoch defined in the state: 2 vs 5. "
        r"Please, .+ "
        r"before calling engine.run\(\) in order to restart the training from the beginning.",
    ):
        state = engine.run(data, max_epochs=2)
    state.max_epochs = None
    engine.run(data, max_epochs=2)
