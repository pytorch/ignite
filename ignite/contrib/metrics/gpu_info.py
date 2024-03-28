""" ``ignite.contrib.metrics.gpu_info`` was moved to ``ignite.metrics.gpu_info``.
Note:
    ``ignite.contrib.metrics.gpu_info`` was moved to ``ignite.metrics.gpu_info``.
    Please refer to :mod:`~ignite.metrics.gpu_info`.
"""

import warnings

removed_in = "0.6.0"
deprecation_warning = (
    f"{__file__} has been moved to ignite/metrics/gpu_info.py"
    + (f" and will be removed in version {removed_in}" if removed_in else "")
    + ".\n Please refer to the documentation for more details."
)
warnings.warn(deprecation_warning, DeprecationWarning, stacklevel=2)
from ignite.metrics.gpu_info import GpuInfo

__all__ = [
    "GpuInfo",
]

GpuInfo = GpuInfo
