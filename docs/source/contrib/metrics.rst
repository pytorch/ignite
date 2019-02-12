ignite.contrib.metrics
======================

Contribution module of metrics

.. currentmodule:: ignite.contrib.metrics

.. automodule:: ignite.contrib.metrics
   :members:
   :imported-members:


Regression metrics
------------------

Module :mod:`ignite.contrib.metrics.regression` provides implementations of
metrics useful for regression tasks. Definitions of metrics are based on `Botchkarev 2018`_, page 30 "Appendix 2. Metrics mathematical definitions".

.. _`Botchkarev 2018`:
        https://arxiv.org/ftp/arxiv/papers/1809/1809.03006.pdf

Complete list of metrics:

    - :class:`~ignite.contrib.metrics.regression.CanberraMetric`
    - :class:`~ignite.contrib.metrics.regression.FractionalAbsoluteError`
    - :class:`~ignite.contrib.metrics.regression.FractionalBias`
    - :class:`~ignite.contrib.metrics.regression.GeometricMeanAbsoluteError`
    - :class:`~ignite.contrib.metrics.regression.GeometricMeanRelativeAbsoluteError`
    - :class:`~ignite.contrib.metrics.regression.ManhattanDistance`
    - :class:`~ignite.contrib.metrics.regression.MaximumAbsoluteError`
    - :class:`~ignite.contrib.metrics.regression.MeanAbsoluteRelativeError`
    - :class:`~ignite.contrib.metrics.regression.MeanError`
    - :class:`~ignite.contrib.metrics.regression.MeanNormalizedBias`
    - :class:`~ignite.contrib.metrics.regression.MedianAbsoluteError`
    - :class:`~ignite.contrib.metrics.regression.MedianAbsolutePercentageError`
    - :class:`~ignite.contrib.metrics.regression.MedianRelativeAbsoluteError`
    - :class:`~ignite.contrib.metrics.regression.WaveHedgesDistance`


.. currentmodule:: ignite.contrib.metrics.regression

.. autoclass:: CanberraMetric

.. autoclass:: FractionalAbsoluteError

.. autoclass:: FractionalBias

.. autoclass:: GeometricMeanAbsoluteError

.. autoclass:: GeometricMeanRelativeAbsoluteError

.. autoclass:: ManhattanDistance

.. autoclass:: MaximumAbsoluteError

.. autoclass:: MeanAbsoluteRelativeError

.. autoclass:: MeanError

.. autoclass:: MeanNormalizedBias

.. autoclass:: MedianAbsoluteError

.. autoclass:: MedianAbsolutePercentageError

.. autoclass:: MedianRelativeAbsoluteError

.. autoclass:: WaveHedgesDistance
