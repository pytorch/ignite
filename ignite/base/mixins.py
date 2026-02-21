from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping


class ResettableHandler(metaclass=ABCMeta):
    """Interface for handlers whose internal state can be reset.

    Subclasses must implement the :meth:`reset` method to clear any accumulated
    state, typically at the beginning of a training run.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the handler's internal state."""
        pass


class Serializable:
    _state_dict_all_req_keys: tuple = ()
    _state_dict_one_of_opt_keys: tuple = ()

    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Mapping) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Argument state_dict should be a dictionary, but given {type(state_dict)}")

        for k in self._state_dict_all_req_keys:
            if k not in state_dict:
                raise ValueError(
                    f"Required state attribute '{k}' is absent in provided state_dict '{state_dict.keys()}'"
                )
        opts = [k in state_dict for k in self._state_dict_one_of_opt_keys]
        if len(opts) > 0 and ((not any(opts)) or (all(opts))):
            raise ValueError(f"state_dict should contain only one of '{self._state_dict_one_of_opt_keys}' keys")
