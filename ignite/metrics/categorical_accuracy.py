from __future__ import division
import warnings

from ignite.metrics.accuracy import Accuracy


class CategoricalAccuracy(Accuracy):
    """
    Note: This metric is deprecated in favor of Accuracy.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("The use of ignite.metrics.CategoricalAccuracy is deprecated, it will be "
                      "removed in 0.2.0. Please use ignite.metrics.Accuracy instead.", DeprecationWarning)
        super(Accuracy, self).__init__(*args, **kwargs)
