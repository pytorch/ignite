""" ``ignite.contrib.metrics.roc_auc`` was moved to ``ignite.metrics.roc_auc``.
Note:
    ``ignite.contrib.metrics.roc_auc`` was moved to ``ignite.metrics.roc_auc``.
    Please refer to :mod:`~ignite.metrics.roc_auc`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to ignite/metrics/roc_auc.py"
    + (f" and will be removed in version {removed_in}" if removed_in else "")
    + ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.metrics.roc_auc import ROC_AUC, RocCurve

__all__ = ["RocCurve", "ROC_AUC"]

RocCurve = RocCurve
ROC_AUC = ROC_AUC
