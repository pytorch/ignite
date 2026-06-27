from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ignite.engine import Engine


class ResettableHandler(metaclass=ABCMeta):
    """Interface for handlers whose internal state can be reset.

    Subclasses must implement the :meth:`reset` method to clear any accumulated
    state, typically at the beginning of a training run.

    .. versionadded:: 0.5.4
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the handler's internal state."""
        pass

    @abstractmethod
    def attach(self, engine: "Engine", *args: Any, **kwargs: Any) -> None:
        """Attach the handler to an engine."""
        pass


class Serializable:
    _state_dict_all_req_keys: tuple[str, ...] = ()
    _state_dict_one_of_opt_keys: tuple[tuple[str, ...], ...] = ()

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

        opt_groups = self._state_dict_one_of_opt_keys
        if len(opt_groups) > 0 and isinstance(opt_groups[0], str):
            opt_groups = (opt_groups,)

        # Handle groups of one-of optional keys
        for one_of_opt_keys in opt_groups:
            if len(one_of_opt_keys) == 0:
                raise ValueError(
                    f"Empty group found in '{self.__class__.__name__}._state_dict_one_of_opt_keys'. "
                    "Each group must contain at least one state attribute key."
                )
            opts = [k in state_dict for k in one_of_opt_keys]
            num_present = sum(opts)
            if num_present != 1:
                raise ValueError(f"state_dict should contain exactly one of '{one_of_opt_keys}' keys")
