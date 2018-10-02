Ignite Documentation
====================

:mod:`ignite` is a high-level library to help with training neural networks in PyTorch.

- ignite helps you write compact but full-featured training loops in a few lines of code
- you get a training loop with metrics, early-stopping, model checkpointing and other features without the boilerplate

Below we show a side-by-side comparison of using pure pytorch and using ignite to create a training loop
to train and validate your model with occasional checkpointing:

.. image:: https://raw.githubusercontent.com/pytorch/ignite/master/assets/ignite_vs_bare_pytorch.png
   :target: https://raw.githubusercontent.com/pytorch/ignite/master/assets/ignite_vs_bare_pytorch.png

As you can see, the code is more concise and readable with ignite. Furthermore, adding additional metrics, or
things like early stopping is a breeze in ignite, but can start to rapidly increase the complexity of
your code when "rolling your own" training loop.


Installation
============

From pip:

.. code:: bash

    pip install pytorch-ignite


From conda:

.. code:: bash

    conda install ignite -c pytorch


From source:

You can clone the source from `github <https://github.com/pytorch/ignite>`_ and run

.. code:: bash

    python setup.py install


.. toctree::
   :maxdepth: 2
   :caption: Notes

   concepts
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   engine
   handlers
   metrics
   exceptions

.. toctree::
   :maxdepth: 2
   :caption: Contrib Package Reference
    
   contrib/engines
   contrib/metrics
   contrib/handlers

.. automodule:: ignite
   :members:
