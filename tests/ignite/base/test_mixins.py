import pytest

from ignite.base import Serializable


class ExampleSerializable(Serializable):
    _state_dict_all_req_keys = ("a", "b")
    _state_dict_one_of_opt_keys = (("c", "d"), ("e", "f"))


def test_state_dict():
    s = Serializable()
    with pytest.raises(NotImplementedError):
        s.state_dict()

def test_load_state_dict():

    s = ExampleSerializable()
    with pytest.raises(TypeError, match=r"Argument state_dict should be a dictionary"):
        s.load_state_dict("abc")

    with pytest.raises(ValueError, match=r"is absent in provided state_dict"):
        s.load_state_dict({})

    with pytest.raises(ValueError, match=r"is absent in provided state_dict"):
        s.load_state_dict({"a": 1})

    with pytest.raises(ValueError, match=r"state_dict should contain only one of"):
        s.load_state_dict({"a": 1, "b": 2})

    with pytest.raises(ValueError, match=r"state_dict should contain only one of"):
        s.load_state_dict({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})

    with pytest.raises(ValueError, match=r"state_dict should contain only one of"):
        s.load_state_dict({"a": 1, "b": 2, "c": 3, "e": 5, "f": 5})

    s.state_dict_user_keys.append("alpha")
    with pytest.raises(ValueError, match=r"Required user state attribute"):
        s.load_state_dict({"a": 1, "b": 2, "c": 3, "e": 4})
