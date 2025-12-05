ignite.contrib.metrics
======================

Contrib module metrics
----------------------

.. currentmodule:: ignite.contrib.metrics

.. autosummary::
    :nosignatures:
    :toctree: ../generated

    AveragePrecision
    CohenKappa
    GpuInfo
    PrecisionRecallCurve
    ROC_AUC
    RocCurve

Regression metrics
------------------

.. currentmodule:: ignite.contrib.metrics.regression

.. automodule:: ignite.contrib.metrics.regression


Module :mod:`ignite.contrib.metrics.regression` provides implementations of
metrics useful for regression tasks. Definitions of metrics are based on `Botchkarev 2018`_, page 30 "Appendix 2. Metrics mathematical definitions".

.. _`Botchkarev 2018`:
        https://arxiv.org/abs/1809.03006

Complete list of metrics:

.. currentmodule:: ignite.contrib.metrics.regression

.. autosummary::
    :nosignatures:
    :toctree: ../generated

    CanberraMetric
    FractionalAbsoluteError
    FractionalBias
    GeometricMeanAbsoluteError
    GeometricMeanRelativeAbsoluteError
    ManhattanDistance
    MaximumAbsoluteError
    MeanAbsoluteRelativeError
    MeanError
    MeanNormalizedBias
    MedianAbsoluteError
    MedianAbsolutePercentageError
    MedianRelativeAbsoluteError
    R2Score
    WaveHedgesDistance
