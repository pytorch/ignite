ignite.contrib.metrics
======================

Contribution module of metrics

.. currentmodule:: ignite.contrib.metrics

.. autosummary::
    :nosignatures:
    :autolist:

Contrib metrics
---------------

AveragePrecision
~~~~~~~~~~~~~~~~
.. autoclass:: AveragePrecision

CohenKappa
~~~~~~~~~~
.. autoclass:: CohenKappa

GpuInfo
~~~~~~~
.. autoclass:: GpuInfo

PrecisionRecallCurve
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: PrecisionRecallCurve

ROC_AUC
~~~~~~~
.. autoclass:: ROC_AUC

RocCurve
~~~~~~~~
.. autoclass:: RocCurve

Regression metrics
------------------

.. currentmodule:: ignite.contrib.metrics.regression

.. automodule:: ignite.contrib.metrics.regression


Module :mod:`ignite.contrib.metrics.regression` provides implementations of
metrics useful for regression tasks. Definitions of metrics are based on `Botchkarev 2018`_, page 30 "Appendix 2. Metrics mathematical definitions".

.. _`Botchkarev 2018`:
        https://arxiv.org/ftp/arxiv/papers/1809/1809.03006.pdf

Complete list of metrics:

.. currentmodule:: ignite.contrib.metrics.regression

.. autosummary::
    :nosignatures:
    :autolist:

CanberraMetric
~~~~~~~~~~~~~~
.. autoclass:: CanberraMetric

FractionalAbsoluteError
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: FractionalAbsoluteError

FractionalBias
~~~~~~~~~~~~~~~
.. autoclass:: FractionalBias

GeometricMeanAbsoluteError
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: GeometricMeanAbsoluteError

GeometricMeanRelativeAbsoluteError
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: GeometricMeanRelativeAbsoluteError

ManhattanDistance
~~~~~~~~~~~~~~~~~
.. autoclass:: ManhattanDistance

MaximumAbsoluteError
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MaximumAbsoluteError

MeanAbsoluteRelativeError
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MeanAbsoluteRelativeError

MeanError
~~~~~~~~~
.. autoclass:: MeanError

MeanNormalizedBias
~~~~~~~~~~~~~~~~~~~
.. autoclass:: MeanNormalizedBias

MedianAbsoluteError
~~~~~~~~~~~~~~~~~~~
.. autoclass:: MedianAbsoluteError

MedianAbsolutePercentageError
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MedianAbsolutePercentageError

MedianRelativeAbsoluteError
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MedianRelativeAbsoluteError

R2Score
~~~~~~~
.. autoclass:: R2Score

WaveHedgesDistance
~~~~~~~~~~~~~~~~~~
.. autoclass:: WaveHedgesDistance
