import inspect
from typing import Callable


def _check_signature(fn: Callable, fn_description: str, *args, **kwargs) -> None:
    signature = inspect.signature(fn)
    try:  # try without engine
        signature.bind(*args, **kwargs)
    except TypeError as exc:
        fn_params = list(signature.parameters)
        exception_msg = str(exc)
        passed_params = list(args) + list(kwargs)
        raise ValueError(
            "Error adding {} '{}': "
            "takes parameters {} but will be called with {}"
            "({}).".format(fn, fn_description, fn_params, passed_params, exception_msg)
        )
