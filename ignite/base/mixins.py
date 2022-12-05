from collections import OrderedDict
from collections.abc import Mapping
from typing import Tuple


class Serializable:

    _state_dict_all_req_keys: Tuple = ()
    _state_dict_one_of_opt_keys: Tuple = ()

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
