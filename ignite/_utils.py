from typing import Tuple, Union

# For compatibilty
from ignite.utils import apply_to_tensor, apply_to_type, convert_tensor, to_onehot


def _to_hours_mins_secs(time_taken: Union[float, int]) -> Tuple[int, int, int]:
    """Convert seconds to hours, mins, and seconds."""
    mins, secs = divmod(time_taken, 60)
    hours, mins = divmod(mins, 60)
    return round(hours), round(mins), round(secs)
