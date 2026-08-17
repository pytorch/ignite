Ignite Your Networks!
=====================

:mod:`ignite` is a high-level library to help with training and evaluating neural networks in PyTorch flexibly and transparently.

.. raw:: html

    <a target="_blank" rel="noopener noreferrer"
    href="https://colab.research.google.com/github/pytorch/ignite/blob/master/assets/tldr/teaser.ipynb">
        <img
            src="https://raw.githubusercontent.com/pytorch/ignite/master/assets/tldr/pytorch-ignite-teaser.gif"
            width=655
            height=801
            alt="pytorch-ignite-teaser"
            style="width: auto !important; height: auto !important; max-width: 80% !important;"
        >
    </a>

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


Docker Images
-------------

Using pre-built images
^^^^^^^^^^^^^^^^^^^^^^

Pull a pre-built docker image from `our Docker Hub <https://hub.docker.com/u/pytorchignite>`_ and run it with docker v19.03+.

.. code:: bash

   docker run --gpus all -it -v $PWD:/workspace/project --network=host --shm-size 16G pytorchignite/base:latest


Available pre-built images are :

- ``pytorchignite/base:latest | pytorchignite/hvd-base:latest | pytorchignite/msdp-apex-base:latest``
- ``pytorchignite/apex:latest | pytorchignite/hvd-apex:latest``
- ``pytorchignite/vision:latest | pytorchignite/hvd-vision:latest | pytorchignite/msdp-apex-vision:latest``
- ``pytorchignite/apex-vision:latest | pytorchignite/hvd-apex-vision:latest``
- ``pytorchignite/nlp:latest | pytorchignite/hvd-nlp:latest | pytorchignite/msdp-apex-nlp:latest``
- ``pytorchignite/apex-nlp:latest | pytorchignite/hvd-apex-nlp:latest``

For more details, `check out on GitHub <https://github.com/pytorch/ignite/tree/master/docker>`_.



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

   About us <https://pytorch.org/ignite/master/about.html>
   governance
