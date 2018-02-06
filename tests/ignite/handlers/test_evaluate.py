from mock import MagicMock
from pytest import raises

from ignite.handlers import Evaluate


def test_evaluate_init():
    with raises(ValueError):
        Evaluate(MagicMock(), [], iteration_interval=1, epoch_interval=1)

    with raises(ValueError):
        Evaluate(MagicMock(), [])
