
# For compatibilty
from ignite.utils import convert_tensor, apply_to_tensor, apply_to_type, to_onehot


def _to_hours_mins_secs(time_taken):
    """Convert seconds to hours, mins, and seconds."""
    mins, secs = divmod(time_taken, 60)
    hours, mins = divmod(mins, 60)
    return hours, mins, secs
