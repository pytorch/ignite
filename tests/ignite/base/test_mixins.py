import pytest

from collections import OrderedDict

from ignite.base import Serializable


def test_state_dict():
    s = Serializable()
    with pytest.raises(NotImplementedError):
        s.state_dict()


def test_load_state_dict():
    s = Serializable()
    s.load_state_dict({})


class ExampleSerializable(Serializable):
    _state_dict_all_req_keys = ("a", "b")
    _state_dict_one_of_opt_keys = (("c", "d"), ("e", "f"))

    def __init__(self):
        super().__init__()
        self.data = {}

    def state_dict(self):
        return {"a": 1, "b": 2, "c": 3, "e": 5}


class EngineStyleSerializable(Serializable):
    """Serializable that mimics Engine's key structure."""

    _state_dict_all_req_keys = ("epoch_length",)
    _state_dict_one_of_opt_keys = (("iteration", "epoch"), ("max_epochs", "max_iters"))

    def __init__(self):
        super().__init__()
        self.data = {}

    def state_dict(self):
        result = OrderedDict()
        for key in self._state_dict_all_req_keys:
            if key in self.data:
                result[key] = self.data[key]

        # Add user keys
        for key in self._state_dict_user_keys:
            if key in self.data:
                result[key] = self.data[key]

        return result


def test_load_state_dict_validation():
    """Test the updated load_state_dict validation."""
    s = ExampleSerializable()

    # Test type check
    with pytest.raises(TypeError, match=r"Argument state_dict should be a dictionary"):
        s.load_state_dict("not a dict")

    # Test missing required keys
    with pytest.raises(ValueError, match=r"Required state attribute 'a' is absent"):
        s.load_state_dict({})

    with pytest.raises(ValueError, match=r"Required state attribute 'b' is absent"):
        s.load_state_dict({"a": 1})

    # Test one-of optional keys - missing all
    with pytest.raises(ValueError, match=r"should contain at least one of"):
        s.load_state_dict({"a": 1, "b": 2})

    # Test one-of optional keys - having all from one group
    with pytest.raises(ValueError, match=r"should contain only one of '\('c', 'd'\)'"):
        s.load_state_dict({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})

    # Test one-of optional keys - having all from another group
    with pytest.raises(ValueError, match=r"should contain only one of '\('e', 'f'\)'"):
        s.load_state_dict({"a": 1, "b": 2, "c": 3, "e": 5, "f": 6})

    # Valid state dict
    s.load_state_dict({"a": 1, "b": 2, "c": 3, "e": 5})
    print("Valid state dict loaded successfully")


@pytest.mark.parametrize(
    "all_req, opt_groups, state_dict, expected_error_match",
    [
        # Case 1: Pure empty - should pass
        (("required",), (), {"required": "value"}, None),
        # Case 2: Nested empty group - should raise descriptive ValueError
        (
            ("required",),
            ((),),
            {"required": "value"},
            r"Empty group found in 'ParametrizedSerializable._state_dict_one_of_opt_keys'",
        ),
        # Case 3: Mixed empty and filled - should raise descriptive ValueError
        (
            ("required",),
            ((), ("opt1", "opt2")),
            {"required": "value", "opt1": "val"},
            r"Empty group found in 'ParametrizedSerializable._state_dict_one_of_opt_keys'",
        ),
        # Case 4: Standard one-of group - missing keys
        (
            ("required",),
            (("opt1", "opt2"),),
            {"required": "value"},
            r"should contain at least one of '\('opt1', 'opt2'\)'",
        ),
        # Case 5: Standard one-of group - too many keys
        (
            ("required",),
            (("opt1", "opt2"),),
            {"required": "value", "opt1": "v1", "opt2": "v2"},
            r"should contain only one of '\('opt1', 'opt2'\)'",
        ),
        # Case 6: Standard one-of group - valid
        (("required",), (("opt1", "opt2"),), {"required": "value", "opt1": "v1"}, None),
    ],
)
def test_optional_groups_logic(all_req, opt_groups, state_dict, expected_error_match):
    """Test various combinations of optional groups using parametrization."""

    class ParametrizedSerializable(Serializable):
        _state_dict_all_req_keys = all_req
        _state_dict_one_of_opt_keys = opt_groups

        def state_dict(self):
            return {}

    s = ParametrizedSerializable()
    if expected_error_match:
        # for negative cases where we expect case to fail
        with pytest.raises(ValueError, match=expected_error_match):
            s.load_state_dict(state_dict)
    else:
        # for positive cases
        s.load_state_dict(state_dict)


def test_engine_style_validation():
    """Test validation that mimics Engine usage."""
    s = EngineStyleSerializable()

    # Valid: iteration + max_iters
    s.load_state_dict({"epoch_length": 100, "iteration": 150, "max_iters": 500})

    # Valid: epoch + max_epochs
    s2 = EngineStyleSerializable()
    s2.load_state_dict({"epoch_length": 100, "epoch": 3, "max_epochs": 10})

    # Invalid: both iteration and epoch
    s3 = EngineStyleSerializable()
    with pytest.raises(ValueError, match="should contain only one of.*iteration.*epoch"):
        s3.load_state_dict({"epoch_length": 100, "iteration": 150, "epoch": 3, "max_epochs": 10})

    # Invalid: both max_epochs and max_iters
    s4 = EngineStyleSerializable()
    with pytest.raises(ValueError, match="should contain only one of.*max_epochs.*max_iters"):
        s4.load_state_dict({"epoch_length": 100, "iteration": 150, "max_epochs": 10, "max_iters": 500})


def test_single_option_group():
    """Test group with single option."""

    class SingleOptionSerializable(Serializable):
        _state_dict_all_req_keys = ("base",)
        _state_dict_one_of_opt_keys = (("single",),)

        def state_dict(self):
            return {}

    s = SingleOptionSerializable()

    # Should require the single option
    with pytest.raises(ValueError, match="should contain at least one of"):
        s.load_state_dict({"base": "value"})

    # Should pass with single option
    s.load_state_dict({"base": "value", "single": "option"})


def test_inheritance_overrides():
    """Test that subclasses can override validation rules."""

    class BaseSerializable(Serializable):
        _state_dict_all_req_keys = ("base_req",)
        _state_dict_one_of_opt_keys = (("base_opt1", "base_opt2"),)

        def state_dict(self):
            return {}

    class DerivedSerializable(BaseSerializable):
        _state_dict_all_req_keys = ("derived_req1", "derived_req2")
        _state_dict_one_of_opt_keys = (("derived_opt1", "derived_opt2"),)

    # Base class uses its own rules
    base = BaseSerializable()
    base.load_state_dict({"base_req": "value", "base_opt1": "opt"})

    # Derived class uses overridden rules
    derived = DerivedSerializable()
    with pytest.raises(ValueError, match="Required state attribute.*derived_req1"):
        derived.load_state_dict({"base_req": "value", "base_opt1": "opt"})

    # Valid for derived class
    derived.load_state_dict({"derived_req1": "d1", "derived_req2": "d2", "derived_opt2": "opt"})


def test_complex_grouped_keys():
    """Test grouped optional keys."""
    s = EngineStyleSerializable()

    # Valid with all requirements
    s.load_state_dict({"epoch_length": 100, "iteration": 250, "max_iters": 500})

    # Missing from a group should fail
    s2 = EngineStyleSerializable()
    with pytest.raises(ValueError, match="should contain at least one of.*iteration.*epoch"):
        s2.load_state_dict({"epoch_length": 100, "max_iters": 500})


def test_backwards_compatibility():
    """Test that old style validation still works."""

    class OldStyleSerializable(Serializable):
        _state_dict_all_req_keys = ("req1", "req2")
        # No _state_dict_one_of_opt_keys defined - should default to empty

        def state_dict(self):
            return {}

    s = OldStyleSerializable()

    # Should work with just required keys
    s.load_state_dict({"req1": "r1", "req2": "r2"})

    # Should fail without required keys
    with pytest.raises(ValueError, match="Required state attribute"):
        s.load_state_dict({"req1": "r1"})


def test_complex_scenario():
    """Test complex scenario with multiple groups and user keys."""

    class ComplexSerializable(Serializable):
        _state_dict_all_req_keys = ("base1", "base2")
        _state_dict_one_of_opt_keys = (
            ("pos1", "pos2", "pos3"),
            ("term1", "term2"),
            ("opt1", "opt2", "opt3", "opt4"),
        )

        def state_dict(self):
            return {}

    s = ComplexSerializable()
    # Valid complex state
    s.load_state_dict(
        {
            "base1": "b1",
            "base2": "b2",
            "pos2": "position",
            "term1": "termination",
            "opt3": "option",
        }
    )

    # Missing from one group should fail
    s2 = ComplexSerializable()
    with pytest.raises(ValueError, match="should contain at least one of.*term1.*term2"):
        s2.load_state_dict({"base1": "b1", "base2": "b2", "pos1": "pos", "opt4": "opt"})
