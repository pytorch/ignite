ignite.contrib.metrics
======================

Contribution module of metrics

.. currentmodule:: ignite.contrib.metrics

.. autosummary::
    :nosignatures:
    :autolist:

.. automodule:: ignite.contrib.metrics
   :members:
   :imported-members:


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

.. autoclass:: R2Score

.. autoclass:: WaveHedgesDistance
