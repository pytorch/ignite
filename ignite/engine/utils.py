import inspect
from typing import Any, Callable, Tuple, Union


def _check_signature(fn: Callable, fn_description: str, *args: Any, **kwargs: Any) -> None:
    # if handler with filter, check the handler rather than the decorator
    if hasattr(fn, "_parent"):
        signature = inspect.signature(fn._parent())  # type: ignore[attr-defined]
    else:
        signature = inspect.signature(fn)
    try:  # try without engine
        signature.bind(*args, **kwargs)
    except TypeError as exc:
        fn_params = list(signature.parameters)
        exception_msg = str(exc)
        passed_params = list(args) + list(kwargs)
        raise ValueError(
            f"Error adding {fn} '{fn_description}': "
            f"takes parameters {fn_params} but will be called with {passed_params}"
            f"({exception_msg})."
        )


def _to_hours_mins_secs(time_taken: Union[float, int]) -> Tuple[int, int, int]:
    """Convert seconds to hours, mins, and seconds."""
    mins, secs = divmod(time_taken, 60)
    hours, mins = divmod(mins, 60)
    return round(hours), round(mins), round(secs)
