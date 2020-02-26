__all__ = ["NotComputableError"]


class NotComputableError(RuntimeError):
    """
    Exception class to raise if Metric cannot be computed.
    """
