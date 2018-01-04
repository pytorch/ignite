from pytest import raises

from ignite.handlers import Validate


def test_validate_init():
    with raises(ValueError):
        Validate([], iteration_interval=1, epoch_interval=1)

    with raises(ValueError):
        Validate([])
