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
        # Total keys = required keys + 1 from each optional group (e.g. iteration & max_epochs)
        expected_len = len(engine._state_dict_all_req_keys) + len(engine._state_dict_one_of_opt_keys)
        assert isinstance(sd, Mapping) and len(sd) == expected_len
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
        # Total keys = required keys + 1 from each optional group + user keys
        num_opt_keys = len(engine._state_dict_one_of_opt_keys)
        expected_len = len(engine._state_dict_all_req_keys) + num_opt_keys + len(engine.state_dict_user_keys)
        assert isinstance(sd, Mapping) and len(sd) == expected_len
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
    # Total keys = required keys + 1 from each optional group (e.g. iteration & max_epochs)
    expected_len = len(engine._state_dict_all_req_keys) + len(engine._state_dict_one_of_opt_keys)
    assert isinstance(sd, Mapping) and len(sd) == expected_len
    assert sd["iteration"] == engine.state.iteration == 10 * 100
    assert sd["epoch_length"] == engine.state.epoch_length == 100
    assert sd["max_epochs"] == engine.state.max_epochs == 10


@pytest.mark.parametrize(
    "state_dict, error_type, match",
    [
        ("not a dict", TypeError, r"Argument state_dict should be a dictionary"),
        ({}, ValueError, r"Required state attribute 'epoch_length' is absent"),
        ({"epoch_length": 100}, ValueError, r"Required state attribute 'iteration' is absent"),
        (
            {"epoch_length": 100, "iteration": 10, "max_epochs": 5, "max_iters": 500},
            ValueError,
            r"should contain exactly one of '\('max_epochs', 'max_iters'\)'",
        ),
        ({"epoch": 5, "max_epochs": 3, "epoch_length": 10}, ValueError, r"greater than or equal to the start epoch"),
        (
            {"iteration": 100, "max_iters": 50, "epoch_length": 10},
            ValueError,
            r"greater than or equal to the start iteration",
        ),
        (
            {"iteration": 12, "epoch_length": 120, "max_epochs": 100},
            ValueError,
            r"Required user state attribute 'alpha'",
        ),
        (
            {"iteration": 50, "epoch": 1, "epoch_length": 100, "max_epochs": 10},
            ValueError,
            r"State dictionary should contain only one of 'iteration' or 'epoch' keys",
        ),
        (
            {"max_epochs": 100, "epoch": 2, "epoch_length": None},
            ValueError,
            r"epoch_length must be a positive integer to calculate missing 'iteration' or 'epoch' key.",
        ),
        (
            {"max_epochs": 100, "iteration": 200, "epoch_length": 0},
            ValueError,
            r"epoch_length must be a positive integer to calculate missing 'iteration' or 'epoch' key.",
        ),
    ],
)
def test_load_state_dict_errors(state_dict, error_type, match):
    engine = Engine(lambda e, b: 1)
    if "alpha" in str(match):
        engine.state_dict_user_keys.append("alpha")
    with pytest.raises(error_type, match=match):
        engine.load_state_dict(state_dict)


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

    # Test missing user key
    engine2 = Engine(lambda e, b: 1)
    engine2.state_dict_user_keys.append("alpha")
    with pytest.raises(ValueError, match="Required user state attribute 'alpha' is absent in provided state_dict"):
        engine2.load_state_dict({"max_epochs": 100, "epoch_length": 120, "iteration": 123})


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


@pytest.mark.parametrize(
    "termination_param, value, expected_iters",
    [
        ("max_epochs", 5, 500),
        ("max_iters", 250, 250),
    ],
)
def test_state_dict_termination_variants(termination_param, value, expected_iters):
    """Test state_dict with different termination parameters."""
    engine = Engine(lambda e, b: 1)
    data = range(100)
    engine.run(data, **{termination_param: value})

    sd = engine.state_dict()
    assert "iteration" in sd
    assert "epoch_length" in sd
    assert termination_param in sd
    other_param = "max_iters" if termination_param == "max_epochs" else "max_epochs"
    assert other_param not in sd
    assert sd[termination_param] == value
    assert sd["epoch_length"] == 100
    assert sd["iteration"] == expected_iters


@pytest.mark.parametrize(
    "state_dict, expected_state",
    [
        (
            {"epoch": 2, "max_epochs": 5, "epoch_length": 100},
            {"epoch": 2, "max_epochs": 5, "epoch_length": 100, "iteration": 200},
        ),
        (
            {"iteration": 150, "max_iters": 250, "epoch_length": 100},
            {"iteration": 150, "max_iters": 250, "epoch_length": 100, "epoch": 1},
        ),
        (
            {"iteration": 150, "max_epochs": 3, "epoch_length": 100},
            {"iteration": 150, "max_epochs": 3, "epoch_length": 100, "epoch": 1},
        ),
        (
            {"epoch": 2, "max_iters": 500, "epoch_length": 100},
            {"epoch": 2, "max_iters": 500, "epoch_length": 100, "iteration": 200},
        ),
    ],
)
def test_load_state_dict_termination_variants(state_dict, expected_state):
    """Test load_state_dict with different combinations of progress and termination params."""
    engine = Engine(lambda e, b: 1)
    engine.load_state_dict(state_dict)

    for attr, expected_value in expected_state.items():
        assert getattr(engine.state, attr) == expected_value


def test_save_and_load_with_max_iters():
    """Test saving and loading engine state with max_iters."""
    # Create and run engine with max_iters
    engine1 = Engine(lambda e, b: b)
    data = list(range(20))
    engine1.run(data, max_iters=50, epoch_length=10)

    # Save state
    state_dict = engine1.state_dict()
    assert state_dict["iteration"] == 50
    assert state_dict["max_iters"] == 50
    assert state_dict["epoch_length"] == 10
    assert "max_epochs" not in state_dict

    # Load state in new engine
    engine2 = Engine(lambda e, b: b)
    engine2.load_state_dict(state_dict)

    assert engine2.state.iteration == 50
    assert engine2.state.max_iters == 50
    assert engine2.state.epoch_length == 10
    assert engine2.state.epoch == 5  # 50 // 10


def test_resume_with_max_iters():
    """Test resuming engine run with max_iters using early termination."""
    counter = [0]

    def update_fn(engine, batch):
        counter[0] += 1
        return batch

    engine = Engine(update_fn)
    data = list(range(10))

    # Set up early termination at iteration 15
    @engine.on(Events.ITERATION_COMPLETED(once=15))
    def stop_early(engine):
        engine.terminate()

    # Run with max_iters=25 but terminate early at 15
    engine.run(data, max_iters=25, epoch_length=10)
    assert counter[0] == 15
    assert engine.state.iteration == 15
    assert engine.state.max_iters == 25  # Still has the original max_iters

    # Save and reload state
    state_dict = engine.state_dict()
    counter[0] = 0  # Reset counter

    engine2 = Engine(update_fn)
    engine2.load_state_dict(state_dict)

    # Resume running - should continue from iteration 15 to 25
    engine2.run(data)
    assert counter[0] == 10  # 25 - 15
    assert engine2.state.iteration == 25


def test_mutually_exclusive_max_epochs_max_iters():
    """Test that max_epochs and max_iters are mutually exclusive."""
    engine = Engine(lambda e, b: 1)
    data = range(10)

    with pytest.raises(ValueError, match="mutually exclusive"):
        engine.run(data, max_epochs=5, max_iters=50)


def test_unknown_epoch_length_with_max_iters():
    """Test handling unknown epoch_length with max_iters."""
    counter = [0]

    def update_fn(engine, batch):
        counter[0] += 1
        return batch

    def data_iter():
        for i in range(15):
            yield i

    engine = Engine(update_fn)

    # Run with unknown epoch length and max_iters that completes before first epoch ends
    engine.run(data_iter(), max_iters=10)
    assert counter[0] == 10
    assert engine.state.iteration == 10
    # epoch_length remains None since we stopped before completing an epoch
    assert engine.state.epoch_length is None

    # State dict should have max_iters
    sd = engine.state_dict()
    assert "max_iters" in sd
    assert sd["max_iters"] == 10

    # Test case where we complete a full epoch
    engine2 = Engine(update_fn)
    counter[0] = 0
    engine2.run(data_iter(), max_iters=20)
    assert counter[0] == 15  # Iterator exhausted after 15
    assert engine2.state.iteration == 15
    # epoch_length should be determined when iterator is exhausted
    assert engine2.state.epoch_length == 15


def test_engine_attributes():
    """Test basic engine attributes and state."""
    engine = Engine(lambda e, b: 1)

    # Test basic attributes exist
    assert hasattr(engine, "state")
    assert hasattr(engine, "logger")
    assert hasattr(engine, "state_dict_user_keys")

    # Test initial state
    assert engine.state.iteration == 0
    assert engine.state.epoch == 0
    assert engine.state.max_epochs is None
    assert engine.state.max_iters is None
    assert engine.state.epoch_length is None


@pytest.mark.parametrize(
    "param_name, current_val, low_val, high_val",
    [
        ("max_epochs", 3, 2, 5),
        ("max_iters", 30, 25, 40),
    ],
)
def test_helper_methods(param_name, current_val, low_val, high_val):
    """Test the helper validation methods."""
    engine = Engine(lambda e, b: 1)
    data = range(10)

    # Initialize engine state
    engine.run(data, **{param_name: current_val})

    helper_method = getattr(engine, f"_check_and_set_{param_name}")

    # Test too low value
    with pytest.raises(ValueError, match="greater than or equal to the start"):
        helper_method(low_val)

    # Test valid higher value
    helper_method(high_val)
    assert getattr(engine.state, param_name) == high_val


def test_backward_compatibility():
    """Test backward compatibility with old state dicts."""
    engine = Engine(lambda e, b: 1)

    # Old state dict format (with max_epochs)
    old_state_dict = {"iteration": 200, "max_epochs": 5, "epoch_length": 100}

    engine.load_state_dict(old_state_dict)
    assert engine.state.iteration == 200
    assert engine.state.max_epochs == 5
    assert engine.state.epoch_length == 100
    assert engine.state.epoch == 2  # 200 // 100


def test_user_keys_with_max_iters():
    """Test user-defined keys work with max_iters."""
    engine = Engine(lambda e, b: b)
    data = list(range(10))

    # Add user keys
    engine.state_dict_user_keys.append("custom_value")
    engine.state_dict_user_keys.append("another_value")

    @engine.on(Events.STARTED)
    def init_custom_values(engine):
        engine.state.custom_value = 42
        engine.state.another_value = "test"

    engine.run(data, max_iters=5)

    # Check state dict contains user keys
    sd = engine.state_dict()
    assert "custom_value" in sd
    assert "another_value" in sd
    assert sd["custom_value"] == 42
    assert sd["another_value"] == "test"
    assert "max_iters" in sd
    assert "max_epochs" not in sd

    # Load into new engine
    engine2 = Engine(lambda e, b: b)
    engine2.state_dict_user_keys.append("custom_value")
    engine2.state_dict_user_keys.append("another_value")

    engine2.load_state_dict(sd)
    assert engine2.state.custom_value == 42
    assert engine2.state.another_value == "test"
    assert engine2.state.max_iters == 5


def test_is_done_method_with_max_iters():
    """Test the _is_done static method with max_iters."""
    # Test with max_iters
    state = State()
    state.iteration = 100
    state.max_iters = 100
    state.epoch_length = 25
    state.epoch = 4
    state.max_epochs = None

    assert Engine._is_done(state) is True

    state.iteration = 99
    assert Engine._is_done(state) is False

    state.iteration = 101
    assert Engine._is_done(state) is True

    # Test with both set (shouldn't happen but test logic)
    state.iteration = 50
    state.max_iters = 100
    state.max_epochs = 3
    state.epoch = 2
    state.epoch_length = 25
    assert Engine._is_done(state) is False

    state.iteration = 100
    assert Engine._is_done(state) is True

    state.iteration = 75
    state.epoch = 3
    assert Engine._is_done(state) is True


def test_none_data_with_max_iters():
    """Test running with None data and max_iters."""
    counter = [0]

    def update_fn(engine, batch):
        assert batch is None
        counter[0] += 1
        return 1

    engine = Engine(update_fn)

    # Should work with None data if epoch_length provided
    engine.run(data=None, max_iters=30, epoch_length=10)

    assert counter[0] == 30
    assert engine.state.iteration == 30
    assert engine.state.max_iters == 30
    assert engine.state.epoch_length == 10
    assert engine.state.epoch == 3  # ceil(30/10) = 3


def test_epoch_calculation_with_max_iters():
    """Test epoch calculation when using max_iters."""
    engine = Engine(lambda e, b: b)
    data = list(range(25))

    # Run with max_iters that doesn't divide evenly
    engine.run(data, max_iters=60)

    assert engine.state.iteration == 60
    assert engine.state.max_iters == 60
    assert engine.state.epoch_length == 25
    assert engine.state.epoch == 3  # ceil(60/25) = 3

    # Save and verify state dict
    sd = engine.state_dict()
    assert sd["iteration"] == 60
    assert sd["max_iters"] == 60
    assert sd["epoch_length"] == 25


def test_resume_with_higher_max_iters():
    """Test loading state and running with higher max_iters value."""
    counter = [0]

    def update_fn(engine, batch):
        counter[0] += 1
        return batch

    # First engine: run until 15 then save state
    engine1 = Engine(update_fn)
    data = list(range(10))

    # Use early termination to simulate partial run
    @engine1.on(Events.ITERATION_COMPLETED(once=15))
    def stop_early(engine):
        engine.terminate()

    engine1.run(data, max_iters=20)
    assert counter[0] == 15
    assert engine1.state.iteration == 15
    assert engine1.state.max_iters == 20

    # Save state
    sd = engine1.state_dict()
    counter[0] = 0

    # Second engine: load state and increase max_iters
    engine2 = Engine(update_fn)
    engine2.load_state_dict(sd)

    # Directly set higher max_iters and run
    engine2.run(data, max_iters=25)
    assert counter[0] == 10  # 25 - 15
    assert engine2.state.iteration == 25
    assert engine2.state.max_iters == 25

    # Final state dict
    final_sd = engine2.state_dict()
    assert final_sd["iteration"] == 25
    assert final_sd["max_iters"] == 25


def test_checkpoint_with_max_iters():
    import tempfile
    import os
    from ignite.handlers import Checkpoint, DiskSaver
    from ignite.engine import Engine, Events

    with tempfile.TemporaryDirectory() as tmpdir:
        data = list(range(10))

        def update_fn(engine, batch):
            return 1

        engine1 = Engine(update_fn)

        # Save after 15 iterations (mid 2nd epoch)
        to_save = {"engine": engine1}
        handler = Checkpoint(to_save, DiskSaver(tmpdir, require_empty=False), n_saved=1)
        engine1.add_event_handler(Events.ITERATION_COMPLETED(once=15), handler)

        @engine1.on(Events.ITERATION_COMPLETED(once=15))
        def stop_early():
            engine1.terminate()

        engine1.run(data, max_iters=25)

        assert engine1.state.iteration == 15
        assert engine1.state.max_iters == 25

        # Reload checkpoint
        engine2 = Engine(update_fn)
        checkpoint_path = os.path.join(tmpdir, os.listdir(tmpdir)[0])
        import torch

        checkpoint = torch.load(checkpoint_path)
        Checkpoint.load_objects(to_load={"engine": engine2}, checkpoint=checkpoint)

        assert engine2.state.iteration == 15
        assert engine2.state.max_iters == 25
        assert engine2.state.epoch_length == 10

        # Resume
        engine2.run(data, max_iters=25)
        assert engine2.state.iteration == 25
        assert engine2.state.epoch == 3
        assert getattr(engine2.state, "max_epochs", None) is None
