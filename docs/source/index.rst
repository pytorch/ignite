Ignite Your Networks!
=====================

:mod:`ignite` is a high-level library to help with training and evaluating neural networks in PyTorch flexibly and transparently.



Library structure
=================

-   :mod:`ignite`: Core of the library, contains an engine for training and
    evaluating, most of the classic machine learning metrics and a
    variety of handlers to ease the pain of training and validation of
    neural networks.

-   :mod:`ignite.contrib`: The contrib directory contains additional
    modules that can require extra dependencies. Modules vary from TBPTT engine,
    various optimisation parameter schedulers, experiment tracking system handlers and a
    metrics module containing many regression metrics.


.. automodule:: ignite

.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   engine
   handlers
   metrics
   distributed
   exceptions
   utils


.. automodule:: ignite.contrib

.. toctree::
   :maxdepth: 2
   :caption: Contrib Package Reference

   contrib/engines
   contrib/metrics
   contrib/handlers


.. toctree::
   :maxdepth: 1
   :caption: Team

   governance
