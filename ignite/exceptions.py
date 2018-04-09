class NotComputableError(RuntimeError):
    """
    Exception class to raise if Metric cannot be computed.
    """


class UndefinedMetricWarning(UserWarning):
    """
    Warning class used if Metric computation is not well-defined.
    """
