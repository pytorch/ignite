from ignite.base import Serializable


def test_load_state_dict():

    s = Serializable()
    s.load_state_dict({})
