#!/usr/bin/env python
"""Test script to verify max_iters handling in Engine state dict.

This tests the fix for issue #1521.
"""
import pytest
from ignite.engine import Engine, Events, State


def test_state_dict_with_max_epochs():
    """Test state_dict with max_epochs set."""
    engine = Engine(lambda e, b: 1)
    data = range(100)
    engine.run(data, max_epochs=5)

    sd = engine.state_dict()
    assert "iteration" in sd
    assert "epoch_length" in sd
    assert "max_epochs" in sd
    assert "max_iters" not in sd
    assert sd["max_epochs"] == 5
    assert sd["epoch_length"] == 100
    assert sd["iteration"] == 500


def test_state_dict_with_max_iters():
    """Test state_dict with max_iters set."""
    engine = Engine(lambda e, b: 1)
    data = range(100)
    engine.run(data, max_iters=250)

    sd = engine.state_dict()
    assert "iteration" in sd
    assert "epoch_length" in sd
    assert "max_iters" in sd
    assert "max_epochs" not in sd
    assert sd["max_iters"] == 250
    assert sd["epoch_length"] == 100
    assert sd["iteration"] == 250


def test_load_state_dict_with_max_epochs():
    """Test load_state_dict with max_epochs."""
    engine = Engine(lambda e, b: 1)

    state_dict = {"epoch": 2, "max_epochs": 5, "epoch_length": 100}

    engine.load_state_dict(state_dict)
    assert engine.state.epoch == 2
    assert engine.state.max_epochs == 5
    assert engine.state.epoch_length == 100
    assert engine.state.iteration == 200


def test_load_state_dict_with_max_iters():
    """Test load_state_dict with max_iters."""
    engine = Engine(lambda e, b: 1)

    state_dict = {"iteration": 150, "max_iters": 250, "epoch_length": 100}

    engine.load_state_dict(state_dict)
    assert engine.state.iteration == 150
    assert engine.state.max_iters == 250
    assert engine.state.epoch_length == 100
    assert engine.state.epoch == 1  # 150 // 100


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


def test_validation_errors():
    """Test validation errors for invalid states."""
    engine = Engine(lambda e, b: 1)

    # Test invalid max_epochs in state_dict
    with pytest.raises(ValueError, match="larger than or equal to the current epoch"):
        state_dict = {"epoch": 5, "max_epochs": 3, "epoch_length": 10}
        engine.load_state_dict(state_dict)

    # Test invalid max_iters in state_dict
    with pytest.raises(ValueError, match="larger than or equal to the current iteration"):
        state_dict = {"iteration": 100, "max_iters": 50, "epoch_length": 10}
        engine.load_state_dict(state_dict)


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


def test_helper_methods():
    """Test the helper validation methods."""
    engine = Engine(lambda e, b: 1)
    data = range(10)
    engine.run(data, max_epochs=3)

    # Test _check_and_set_max_epochs
    with pytest.raises(ValueError, match="greater than or equal to the start"):
        engine._check_and_set_max_epochs(2)

    engine._check_and_set_max_epochs(5)
    assert engine.state.max_epochs == 5

    # Test _check_and_set_max_iters
    engine.state.max_epochs = None
    engine.state.max_iters = 30

    with pytest.raises(ValueError, match="greater than or equal to the start"):
        engine._check_and_set_max_iters(25)

    engine._check_and_set_max_iters(40)
    assert engine.state.max_iters == 40


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


def test_invalid_state_dict_both_termination_params():
    """Test that state dict with both max_epochs and max_iters fails."""
    engine = Engine(lambda e, b: 1)

    state_dict = {"iteration": 100, "max_epochs": 5, "max_iters": 500, "epoch_length": 100}

    with pytest.raises(ValueError, match="should contain only one of"):
        engine.load_state_dict(state_dict)


def test_invalid_state_dict_both_position_params():
    """Test that state dict with both iteration and epoch fails."""
    engine = Engine(lambda e, b: 1)

    state_dict = {"iteration": 100, "epoch": 2, "max_epochs": 5, "epoch_length": 100}

    with pytest.raises(ValueError, match="should contain only one of"):
        engine.load_state_dict(state_dict)


def test_invalid_state_dict_missing_termination():
    """Test that state dict without max_epochs or max_iters fails."""
    engine = Engine(lambda e, b: 1)

    state_dict = {"iteration": 100, "epoch_length": 100}

    with pytest.raises(ValueError, match="should contain at least one of"):
        engine.load_state_dict(state_dict)


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
    engine2.state.max_iters = 25
    engine2.run(data)
    assert counter[0] == 10  # 25 - 15
    assert engine2.state.iteration == 25
    assert engine2.state.max_iters == 25

    # Final state dict
    final_sd = engine2.state_dict()
    assert final_sd["iteration"] == 25
    assert final_sd["max_iters"] == 25
