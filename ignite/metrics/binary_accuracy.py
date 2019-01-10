from __future__ import division
import warnings

from ignite.metrics.accuracy import Accuracy


class BinaryAccuracy(Accuracy):
    """
    Note: This metric is deprecated in favor of :class:`~ignite.metrics.Accuracy`.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("The use of ignite.metrics.BinaryAccuracy is deprecated, it will be "
                      "removed in 0.2.0. Please use ignite.metrics.Accuracy instead.", DeprecationWarning)
        super(Accuracy, self).__init__(*args, **kwargs)
