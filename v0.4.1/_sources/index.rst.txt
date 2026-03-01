Ignite Your Networks!
=====================

:mod:`ignite` is a high-level library to help with training and evaluating neural networks in PyTorch flexibly and transparently.

.. image:: https://raw.githubusercontent.com/pytorch/ignite/master/assets/tldr/pytorch-ignite-teaser.gif
   :width: 460
   :target: https://raw.githubusercontent.com/pytorch/ignite/master/assets/tldr/teaser.py

*Click on the image to see complete code*


Features
--------

- `Less code than pure PyTorch <https://raw.githubusercontent.com/pytorch/ignite/master/assets/ignite_vs_bare_pytorch.png>`_ while ensuring maximum control and simplicity

- Library approach and no program's control inversion - *Use ignite where and when you need*

- Extensible API for metrics, experiment managers, and other components


Installation
============

From `pip <https://pypi.org/project/pytorch-ignite/>`_:

.. code:: bash

    pip install pytorch-ignite


From `conda <https://anaconda.org/pytorch/ignite>`_:

.. code:: bash

    conda install ignite -c pytorch


From source:

.. code:: bash

    pip install git+https://github.com/pytorch/ignite



Nightly releases
----------------

From pip:

.. code:: bash

    pip install --pre pytorch-ignite


From conda (this suggests to install `pytorch nightly release <https://anaconda.org/pytorch-nightly/pytorch>`_ instead
of stable version as dependency):

.. code:: bash

    conda install ignite -c pytorch-nightly


Documentation
=============

To get started, please, read :doc:`quickstart` and :doc:`concepts`.

.. toctree::
   :maxdepth: 2
   :caption: Notes

   quickstart
   concepts
   examples
   faq


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

   about
   governance
