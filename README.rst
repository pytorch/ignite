Ignite
======

.. image:: https://travis-ci.org/pytorch/ignite.svg?branch=master
    :target: https://travis-ci.org/pytorch/ignite

.. image:: https://codecov.io/gh/pytorch/ignite/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/pytorch/ignite

.. image:: https://pepy.tech/badge/pytorch-ignite
    :target: https://pepy.tech/project/pytorch-ignite

.. image:: https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpytorch-ignite%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v
    :target: https://pytorch.org/ignite/index.html
    
Ignite is a high-level library to help with training neural networks in PyTorch.

- ignite helps you write compact but full-featured training loops in a few lines of code
- you get a training loop with metrics, early-stopping, model checkpointing and other features without the boilerplate

Below we show a side-by-side comparison of using pure pytorch and using ignite to create a training loop
to train and validate your model with occasional checkpointing:

.. image:: assets/ignite_vs_bare_pytorch.png
    :target: https://raw.githubusercontent.com/pytorch/ignite/master/assets/ignite_vs_bare_pytorch.png

As you can see, the code is more concise and readable with ignite. Furthermore, adding additional metrics, or
things like early stopping is a breeze in ignite, but can start to rapidly increase the complexity of
your code when "rolling your own" training loop.


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


Why Ignite?
===========
Ignite's high level of abstraction assumes less about the type of network (or networks) that you are training, and we require the user to define the closure to be run in the training and validation loop. This level of abstraction allows for a great deal more of flexibility, such as co-training multiple models (i.e. GANs) and computing/tracking multiple losses and metrics in your training loop.

Ignite also allows for multiple handlers to be attached to events, and a finer granularity of events in the engine loop.


Documentation
=============
API documentation and an overview of the library can be found `here <https://pytorch.org/ignite/index.html>`_.


Structure
=========
- **ignite**: Core of the library, contains an engine for training and evaluating, all of the classic machine learning metrics and a variety of handlers to ease the pain of training and validation of neural networks! 

- **ignite.contrib**: The Contrib directory contains additional modules contributed by Ignite users. Modules vary from TBPTT engine, various optimisation parameter schedulers, logging handlers and a metrics module containing many regression metrics (`ignite.contrib.metrics.regression <https://github.com/pytorch/ignite/tree/master/ignite/contrib/metrics/regression>`_)! 

The code in **ignite.contrib** is not as fully maintained as the core part of the library. It may change or be removed at any time without notice.


Examples
========

We provide several examples ported from `pytorch/examples <https://github.com/pytorch/examples>`_ using `ignite`
to display how it helps to write compact and full-featured training loops in a few lines of code:

MNIST example
--------------

Basic neural network training on MNIST dataset with/without `ignite.contrib` module:

- `MNIST with ignite.contrib TQDM/Tensorboard/Visdom loggers <https://github.com/pytorch/ignite/tree/master/examples/contrib/mnist>`_
- `MNIST with native TQDM/Tensorboard/Visdom logging <https://github.com/pytorch/ignite/tree/master/examples/mnist>`_

Distributed CIFAR10 example
---------------------------

Training a small variant of ResNet on CIFAR10 in various configurations: 1) single gpu, 2) single node multiple gpus, 3) multiple nodes and multilple gpus.

- `CIFAR10 <https://github.com/pytorch/ignite/tree/master/examples/contrib/cifar10>`_


Other examples
--------------

- `DCGAN <https://github.com/pytorch/ignite/tree/master/examples/gan>`_
- `Reinforcement Learning <https://github.com/pytorch/ignite/tree/master/examples/reinforcement_learning>`_
- `Fast Neural Style <https://github.com/pytorch/ignite/tree/master/examples/fast_neural_style>`_


Notebooks
---------

- `Text Classification using Convolutional Neural Networks <https://github.com/pytorch/ignite/blob/master/examples/notebooks/TextCNN.ipynb>`_
- `Variational Auto Encoders <https://github.com/pytorch/ignite/blob/master/examples/notebooks/VAE.ipynb>`_
- `Training Cycle-GAN on Horses to Zebras <https://github.com/pytorch/ignite/blob/master/examples/notebooks/CycleGAN.ipynb>`_
- `Finetuning EfficientNet-B0 on CIFAR100 <https://github.com/pytorch/ignite/blob/master/examples/notebooks/EfficientNet_Cifar100_finetuning.ipynb>`_
- `Convolutional Neural Networks for Classifying Fashion-MNIST Dataset <https://github.com/pytorch/ignite/blob/master/examples/notebooks/FashionMNIST.ipynb>`_


Contributing
============
We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion. If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.

Please see the `contribution guidelines <https://github.com/pytorch/ignite/blob/master/CONTRIBUTING.md>`_ for more information.

As always, PRs are welcome :)
