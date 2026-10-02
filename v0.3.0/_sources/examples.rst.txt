Examples
========

We provide several examples ported from `pytorch/examples <https://github.com/pytorch/examples>`_ using `ignite`
to display how it helps to write compact and full-featured training loops in a few lines of code:

MNIST example
-------------

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
- `Hyperparameters tuning with Ax <https://github.com/pytorch/ignite/blob/master/examples/notebooks/Cifar10_Ax_hyperparam_tuning.ipynb>`_


Reproducible trainings
----------------------

Inspired by `torchvision/references <https://github.com/pytorch/vision/tree/master/references>`_, we provide several
reproducible baselines for vision tasks:

- `ImageNet <https://github.com/pytorch/ignite/blob/master/examples/references/classification/imagenet>`_
- `Pascal VOC2012 <https://github.com/pytorch/ignite/blob/master/examples/references/segmentation/pascal_voc2012>`_

Features:

- Distributed training with mixed precision by `nvidia/apex <https://github.com/NVIDIA/apex/>`_
- Experiments tracking with `MLflow <https://mlflow.org/>`_ or `Polyaxon <https://polyaxon.com/>`_
