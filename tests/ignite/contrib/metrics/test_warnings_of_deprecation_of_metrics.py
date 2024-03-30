from importlib import __import__

import pytest


@pytest.mark.parametrize(
    "log_module,fromlist",
    [
        ("average_precision", ["AveragePrecision"]),
        ("cohen_kappa", ["CohenKappa"]),
        ("gpu_info", ["GpuInfo"]),
        ("precision_recall_curve", ["PrecisionRecallCurve"]),
        ("roc_auc", ["ROC_AUC", "RocCurve"]),
        ("regression.canberra_metric", ["CanberraMetric"]),
        ("regression.fractional_absolute_error", ["FractionalAbsoluteError"]),
        ("regression.fractional_bias", ["FractionalBias"]),
        ("regression.geometric_mean_absolute_error", ["GeometricMeanAbsoluteError"]),
        ("regression.geometric_mean_relative_absolute_error", ["GeometricMeanRelativeAbsoluteError"]),
        ("regression.manhattan_distance", ["ManhattanDistance"]),
        ("regression.maximum_absolute_error", ["MaximumAbsoluteError"]),
        ("regression.mean_absolute_relative_error", ["MeanAbsoluteRelativeError"]),
        ("regression.mean_error", ["MeanError"]),
        ("regression.mean_normalized_bias", ["MeanNormalizedBias"]),
        ("regression.median_absolute_error", ["MedianAbsoluteError"]),
        ("regression.median_absolute_percentage_error", ["MedianAbsolutePercentageError"]),
        ("regression.median_relative_absolute_error", ["MedianRelativeAbsoluteError"]),
        ("regression.r2_score", ["R2Score"]),
        ("regression.wave_hedges_distance", ["WaveHedgesDistance"]),
    ],
)
def test_imports(log_module, fromlist):
    with pytest.warns(DeprecationWarning, match="will be removed in version 0.6.0"):
        imported = __import__(f"ignite.contrib.metrics.{log_module}", globals(), locals(), fromlist)
        for attr in fromlist:
            getattr(imported, attr)
