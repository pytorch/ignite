from ignite.base import Serializable


def test_state_dict_and_load_state_dict():

    s = Serializable()
    s.state_dict()
    s.load_state_dict({})
