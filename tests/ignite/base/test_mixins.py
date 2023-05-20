import pytest

from ignite.base import Serializable


def test_state_dict():
    s = Serializable()
    with pytest.raises(NotImplementedError):
        s.state_dict()


def test_load_state_dict():
    s = Serializable()
    s.load_state_dict({})
