<div align="center">

![Ignite Logo](assets/ignite_logo.svg)


[![image](https://travis-ci.org/pytorch/ignite.svg?branch=master)](https://travis-ci.org/pytorch/ignite)
[![image](https://github.com/pytorch/ignite/workflows/.github/workflows/unittests.yml/badge.svg?branch=master)](https://github.com/pytorch/ignite/actions)
[![image](https://codecov.io/gh/pytorch/ignite/branch/master/graph/badge.svg)](https://codecov.io/gh/pytorch/ignite)
[![image](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpytorch-ignite%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://pytorch.org/ignite/index.html)


[![image](https://anaconda.org/pytorch/ignite/badges/version.svg)](https://anaconda.org/pytorch/ignite)
[![image](https://anaconda.org/pytorch/ignite/badges/downloads.svg)](https://anaconda.org/pytorch/ignite)
[![image](https://img.shields.io/badge/dynamic/json.svg?label=PyPI&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpytorch-ignite%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://pypi.org/project/pytorch-ignite/)
[![image](https://pepy.tech/badge/pytorch-ignite)](https://pepy.tech/project/pytorch-ignite)

[![image](https://img.shields.io/badge/Optuna-integrated-blue)](https://optuna.org)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## TL;DR

Ignite is a high-level library to help with training neural networks in
PyTorch.

-   ignite helps you write compact but full-featured training loops in a
    few lines of code
-   you get a training loop with metrics, early-stopping, model
    checkpointing and other features without the boilerplate

Below we show a side-by-side comparison of using pure pytorch and using
ignite to create a training loop to train and validate your model with
occasional checkpointing:

[![image](assets/ignite_vs_bare_pytorch.png)](https://raw.githubusercontent.com/pytorch/ignite/master/assets/ignite_vs_bare_pytorch.png)

As you can see, the code is more concise and readable with ignite.
Furthermore, adding additional metrics, or things like early stopping is
a breeze in ignite, but can start to rapidly increase the complexity of
your code when \"rolling your own\" training loop.

# Table of Contents
- [Installation](#installation)
  * [Nightly releases](#nightly-releases)
- [Why Ignite?](#why-ignite)
- [Documentation](#documentation)
  * [Additional Materials](#additional-materials)
- [Structure](#structure)
- [Examples](#examples)
  * [MNIST Example](#mnist-example)
  * [Tutorials](#tutorials)
  * [Distributed CIFAR10 Example](#distributed-cifar10-example)
  * [Other Examples](#other-examples)
  * [Reproducible Training Examples](#reproducible-training-examples)
- [Contributing](#contributing)
- [Projects using Ignite](#projects-using-ignite)
- [User feedback](#user-feedback)


# Installation

From [pip](https://pypi.org/project/pytorch-ignite/):

``` {.sourceCode .bash}
pip install pytorch-ignite
```

From [conda](https://anaconda.org/pytorch/ignite):

``` {.sourceCode .bash}
conda install ignite -c pytorch
```

From source:

``` {.sourceCode .bash}
pip install git+https://github.com/pytorch/ignite
```

## Nightly releases

From pip:

``` {.sourceCode .bash}
pip install --pre pytorch-ignite
```

From conda (this suggests to install [pytorch nightly
release](https://anaconda.org/pytorch-nightly/pytorch) instead of stable
version as dependency):

``` {.sourceCode .bash}
conda install ignite -c pytorch-nightly
```

# Why Ignite?

Ignite\'s high level of abstraction assumes less about the type of
network (or networks) that you are training, and we require the user to
define the closure to be run in the training and validation loop. This
level of abstraction allows for a great deal more of flexibility, such
as co-training multiple models (i.e. GANs) and computing/tracking
multiple losses and metrics in your training loop.

Ignite also allows for multiple handlers to be attached to events, and a
finer granularity of events in the engine loop.

# Documentation

API documentation and an overview of the library can be found
[here](https://pytorch.org/ignite/index.html).

## Additional Materials

- [8 Creators and Core Contributors Talk About Their Model Training Libraries From PyTorch Ecosystem](https://neptune.ai/blog/model-training-libraries-pytorch-ecosystem?utm_source=reddit&utm_medium=post&utm_campaign=blog-model-training-libraries-pytorch-ecosystem)
- Ignite Posters from Pytorch Developer Conferences:
    - [2019](https://drive.google.com/open?id=1bqIl-EM6GCCCoSixFZxhIbuF25F2qTZg)
    - [2018](https://drive.google.com/open?id=1_2vzBJ0KeCjGv1srojMHiJRvceSVbVR5)



# Structure

-   **ignite**: Core of the library, contains an engine for training and
    evaluating, all of the classic machine learning metrics and a
    variety of handlers to ease the pain of training and validation of
    neural networks!
-   **ignite.contrib**: The Contrib directory contains additional
    modules contributed by Ignite users. Modules vary from TBPTT engine,
    various optimisation parameter schedulers, logging handlers and a
    metrics module containing many regression metrics
    ([ignite.contrib.metrics.regression](https://github.com/pytorch/ignite/tree/master/ignite/contrib/metrics/regression))!

The code in **ignite.contrib** is not as fully maintained as the core
part of the library. It may change or be removed at any time without
notice.

# Examples

We provide several examples ported from
[pytorch/examples](https://github.com/pytorch/examples) using `ignite` to display how it helps to write compact and
full-featured training loops in a few lines of code:

## MNIST Example

Basic neural network training on MNIST dataset with/without `ignite.contrib` module:

-   [MNIST with ignite.contrib TQDM/Tensorboard/Visdom
    loggers](https://github.com/pytorch/ignite/tree/master/examples/contrib/mnist)
-   [MNIST with native TQDM/Tensorboard/Visdom
    logging](https://github.com/pytorch/ignite/tree/master/examples/mnist)

## Tutorials

-   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/TextCNN.ipynb)  [Text Classification using Convolutional Neural
    Networks](https://github.com/pytorch/ignite/blob/master/examples/notebooks/TextCNN.ipynb) 
-   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/VAE.ipynb)  [Variational Auto
    Encoders](https://github.com/pytorch/ignite/blob/master/examples/notebooks/VAE.ipynb) 
-   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/FashionMNIST.ipynb)  [Convolutional Neural Networks for Classifying Fashion-MNIST
    Dataset](https://github.com/pytorch/ignite/blob/master/examples/notebooks/FashionMNIST.ipynb)
-   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/CycleGAN.ipynb)  [Training Cycle-GAN on Horses to
    Zebras](https://github.com/pytorch/ignite/blob/master/examples/notebooks/CycleGAN.ipynb) 
-   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/EfficientNet_Cifar100_finetuning.ipynb)  [Finetuning EfficientNet-B0 on
    CIFAR100](https://github.com/pytorch/ignite/blob/master/examples/notebooks/EfficientNet_Cifar100_finetuning.ipynb)
-   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/Cifar10_Ax_hyperparam_tuning.ipynb)  [Hyperparameters tuning with
    Ax](https://github.com/pytorch/ignite/blob/master/examples/notebooks/Cifar10_Ax_hyperparam_tuning.ipynb) 
-   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/FastaiLRFinder_MNIST.ipynb)  [Basic example of LR finder on MNIST](https://github.com/pytorch/ignite/blob/master/examples/notebooks/FastaiLRFinder_MNIST.ipynb) 

## Distributed CIFAR10 Example

Training a small variant of ResNet on CIFAR10 in various configurations:
1\) single gpu, 2) single node multiple gpus, 3) multiple nodes and
multilple gpus.

-   [CIFAR10](https://github.com/pytorch/ignite/tree/master/examples/contrib/cifar10)

## Other Examples

-   [DCGAN](https://github.com/pytorch/ignite/tree/master/examples/gan)
-   [Reinforcement
    Learning](https://github.com/pytorch/ignite/tree/master/examples/reinforcement_learning)
-   [Fast Neural
    Style](https://github.com/pytorch/ignite/tree/master/examples/fast_neural_style)

## Reproducible Training Examples

Inspired by
[torchvision/references](https://github.com/pytorch/vision/tree/master/references),
we provide several reproducible baselines for vision tasks:

-   [ImageNet](examples/references/classification/imagenet)
-   [Pascal VOC2012](examples/references/segmentation/pascal_voc2012)

Features:

-   Distributed training with mixed precision by
    [nvidia/apex](https://github.com/NVIDIA/apex/)
-   Experiments tracking with [MLflow](https://mlflow.org/) or
    [Polyaxon](https://polyaxon.com/)

# Contributing

We appreciate all contributions. If you are planning to contribute back
bug-fixes, please do so without any further discussion. If you plan to
contribute new features, utility functions or extensions, please first
open an issue and discuss the feature with us.

Please see the [contribution
guidelines](https://github.com/pytorch/ignite/blob/master/CONTRIBUTING.md)
for more information.

As always, PRs are welcome :)

# Projects using Ignite

-   [State-of-the-Art Conversational AI with Transfer
    Learning](https://github.com/huggingface/transfer-learning-conv-ai)
-   [Tutorial on Transfer Learning in NLP held at NAACL
    2019](https://github.com/huggingface/naacl_transfer_learning_tutorial)
-   [Implementation of \"Attention is All You Need\"
    paper](https://github.com/akurniawan/pytorch-transformer)
-   [Implementation of DropBlock: A regularization method for
    convolutional networks in
    PyTorch](https://github.com/miguelvr/dropblock)
-   [Deep-Reinforcement-Learning-Hands-On-Second-Edition, published by
    Packt](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition)
-   [Kaggle Kuzushiji Recognition: 2nd place
    solution](https://github.com/lopuhin/kaggle-kuzushiji-2019)
-   [Unsupervised Data Augmentation experiments in
    PyTorch](https://github.com/vfdev-5/UDA-pytorch)
-   [Hyperparameters tuning with
    Optuna](https://github.com/pfnet/optuna/blob/master/examples/pytorch_ignite_simple.py)
-   [Project MONAI -
    AI Toolkit for Healthcare Imaging
    ](https://github.com/Project-MONAI/MONAI)

See other projects at [\"Used
by\"](https://github.com/pytorch/ignite/network/dependents?package_id=UGFja2FnZS02NzI5ODEwNA%3D%3D)

If your project implements a paper, represents other use-cases not
covered in our official tutorials, Kaggle competition\'s code or just
your code presents interesting results and uses Ignite. We would like to
add your project in this list, so please send a PR with brief
description of the project.

# User feedback

We have created a form for [\"user
feedback\"](https://github.com/pytorch/ignite/issues/new/choose). We
appreciate any type of feedback and this is how we would like to see our
community:

-   If you like the project and want to say thanks, this the right
    place.
-   If you do not like something, please, share it with us and we can
    see how to improve it.

Thank you !
