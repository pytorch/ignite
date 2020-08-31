<div align="center">

![Ignite Logo](assets/ignite_logo.svg)


[![image](https://travis-ci.org/pytorch/ignite.svg?branch=master)](https://travis-ci.org/pytorch/ignite)
[![image](https://github.com/pytorch/ignite/workflows/.github/workflows/unittests.yml/badge.svg?branch=master)](https://github.com/pytorch/ignite/actions)
[![image](https://img.shields.io/badge/-GPU%20tests-black?style=flat-square)](https://circleci.com/gh/pytorch/ignite)[![image](https://circleci.com/gh/pytorch/ignite.svg?style=svg)](https://circleci.com/gh/pytorch/ignite)
[![image](https://codecov.io/gh/pytorch/ignite/branch/master/graph/badge.svg)](https://codecov.io/gh/pytorch/ignite)
[![image](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpytorch-ignite%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://pytorch.org/ignite/index.html)


![image](https://img.shields.io/badge/-Stable%20Releases:-black?style=flat-square) [![image](https://anaconda.org/pytorch/ignite/badges/version.svg)](https://anaconda.org/pytorch/ignite)
[![image](https://anaconda.org/pytorch/ignite/badges/downloads.svg)](https://anaconda.org/pytorch/ignite)
[![image](https://img.shields.io/badge/dynamic/json.svg?label=PyPI&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpytorch-ignite%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://pypi.org/project/pytorch-ignite/)
[![image](https://pepy.tech/badge/pytorch-ignite)](https://pepy.tech/project/pytorch-ignite)

![image](https://img.shields.io/badge/-Nightly%20Releases:-black?style=flat-square) [![image](https://anaconda.org/pytorch-nightly/ignite/badges/version.svg)](https://anaconda.org/pytorch-nightly/ignite)
[![image](https://img.shields.io/badge/PyPI-pre%20releases-brightgreen)](https://pypi.org/project/pytorch-ignite/#history)

![image](https://img.shields.io/badge/-Features:-black?style=flat-square) [![image](https://img.shields.io/badge/Optuna-integrated-blue)](https://optuna.org)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Twitter](https://img.shields.io/badge/news-twitter-blue)](https://twitter.com/pytorch_ignite)

![image](https://img.shields.io/badge/-Supported_PyTorch/Python_versions:-black?style=flat-square) [![link](https://img.shields.io/badge/-check_here-blue)](https://github.com/pytorch/ignite/actions?query=workflow%3A.github%2Fworkflows%2Fpytorch-version-tests.yml)


</div>

## TL;DR

Ignite is a high-level library to help with training and evaluating neural networks in PyTorch flexibly and transparently.

<div align="center">

<a href="https://colab.research.google.com/github/pytorch/ignite/blob/master/assets/tldr/teaser.ipynb">
 <img alt="PyTorch-Ignite teaser" 
      src="assets/tldr/pytorch-ignite-teaser.gif" 
      width=532">
</a> 

*Click on the image to see complete code*
</div>


### Features

- [Less code than pure PyTorch](https://raw.githubusercontent.com/pytorch/ignite/master/assets/ignite_vs_bare_pytorch.png)
while ensuring maximum control and simplicity

- Library approach and no program's control inversion - *Use ignite where and when you need*

- Extensible API for metrics, experiment managers, and other components


<!-- ############################################################################################################### -->


# Table of Contents
- [Why Ignite?](#why-ignite)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [Structure](#structure)
- [Examples](#examples)
  * [Tutorials](#tutorials)
  * [Reproducible Training Examples](#reproducible-training-examples)
- [Communication](#communication)
- [Contributing](#contributing)
- [Projects using Ignite](#projects-using-ignite)
- [About the team](#about-the-team)


<!-- ############################################################################################################### -->


# Why Ignite?

Ignite is a **library** that provides three high-level features:

- Extremely simple engine and event system
- Out-of-the-box metrics to easily evaluate models
- Built-in handlers to compose training pipeline, save artifacts and log parameters and metrics

## Simplified training and validation loop

No more coding `for/while` loops on epochs and iterations. Users instantiate engines and run them. 

<details>
<summary>
Example
</summary>

```python
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Accuracy


# Setup training engine:
def train_step(engine, batch):
    # Users can do whatever they need on a single iteration 
    # E.g. forward/backward pass for any number of models, optimizers etc
    # ...

trainer = Engine(train_step)

# Setup single model evaluation engine
evaluator = create_supervised_evaluator(model, metrics={"accuracy": Accuracy()})

def validation():        
    state = evaluator.run(validation_data_loader)
    # print computed metrics 
    print(trainer.state.epoch, state.metrics)

# Run model's validation at the end of each epoch
trainer.add_event_handler(Events.EPOCH_COMPLETED, validation)

# Start the training
trainer.run(training_data_loader, max_epochs=100)
```

</details>


## Power of Events & Handlers

The cool thing with handlers is that they offer unparalleled flexibility (compared to say, callbacks). Handlers can be any function: e.g. lambda, simple function, class method etc. Thus, we do not require to inherit from an interface and override its abstract methods which could unnecessarily bulk up your code and its complexity.

### Execute any number of functions whenever you wish

<details>
<summary>
Examples
</summary>

```python
trainer.add_event_handler(Events.STARTED, lambda _: print("Start training"))

# attach handler with args, kwargs
mydata = [1, 2, 3, 4]
logger = ...

def on_training_ended(data):
    print("Training is ended. mydata={}".format(data))
    # User can use variables from another scope  
    logger.info("Training is ended")


trainer.add_event_handler(Events.COMPLETED, on_training_ended, mydata)
# call any number of functions on a single event
trainer.add_event_handler(Events.COMPLETED, lambda engine: print(engine.state.times))

@trainer.on(Events.ITERATION_COMPLETED)
def log_something(engine):
    print(engine.state.output)
```

</details>

### Built-in events filtering

<details>
<summary>
Examples
</summary>

```python
# run the validation every 5 epochs
@trainer.on(Events.EPOCH_COMPLETED(every=5))
def run_validation():
    # run validation

# change some training variable once on 20th epoch
@trainer.on(Events.EPOCH_STARTED(once=20))
def change_training_variable():
    # ...

# Trigger handler with customly defined frequency
@trainer.on(Events.ITERATION_COMPLETED(event_filter=first_x_iters))
def log_gradients():
    # ...
```

</details>

### Stack events to share some actions

<details>
<summary>
Examples
</summary>

Events can be stacked together to enable multiple calls:
```python
@trainer.on(Events.COMPLETED | Events.EPOCH_COMPLETED(every=10))
def run_validation():
    # ...
```

</details>

### Custom events to go beyond standard events

<details>
<summary>
Examples
</summary>

Custom events related to backward and optimizer step calls:
```python
class BackpropEvents(EventEnum):
    BACKWARD_STARTED = 'backward_started'
    BACKWARD_COMPLETED = 'backward_completed'
    OPTIM_STEP_COMPLETED = 'optim_step_completed'

def update(engine, batch):
    # ...
    loss = criterion(y_pred, y)
    engine.fire_event(BackpropEvents.BACKWARD_STARTED)
    loss.backward()
    engine.fire_event(BackpropEvents.BACKWARD_COMPLETED)
    optimizer.step()
    engine.fire_event(BackpropEvents.OPTIM_STEP_COMPLETED)
    # ...    

trainer = Engine(update)
trainer.register_events(*BackpropEvents)

@trainer.on(BackpropEvents.BACKWARD_STARTED)
def function_before_backprop(engine):
    # ...
```
- Complete snippet can be found [here](https://pytorch.org/ignite/faq.html#creating-custom-events-based-on-forward-backward-pass).
- Another use-case of custom events: [trainer for Truncated Backprop Through Time](https://pytorch.org/ignite/contrib/engines.html#ignite.contrib.engines.create_supervised_tbptt_trainer).  

</details>

## Out-of-the-box metrics

- [Metrics](https://pytorch.org/ignite/metrics.html#complete-list-of-metrics) for various tasks: 
Precision, Recall, Accuracy, Confusion Matrix, IoU etc, ~20 [regression metrics](https://pytorch.org/ignite/contrib/metrics.html#regression-metrics).

- Users can also [compose their own metrics](https://pytorch.org/ignite/metrics.html#metric-arithmetics) with ease from 
existing ones using arithmetic operations or torch methods.

<details>
<summary>
Example
</summary>

```python
precision = Precision(average=False)
recall = Recall(average=False)
F1_per_class = (precision * recall * 2 / (precision + recall))
F1_mean = F1_per_class.mean()  # torch mean method
F1_mean.attach(engine, "F1")
```

</details>


<!-- ############################################################################################################### -->


# Installation

From [pip](https://pypi.org/project/pytorch-ignite/):

```bash
pip install pytorch-ignite
```

From [conda](https://anaconda.org/pytorch/ignite):

```bash
conda install ignite -c pytorch
```

From source:

```bash
pip install git+https://github.com/pytorch/ignite
```

## Nightly releases

From pip:

```bash
pip install --pre pytorch-ignite
```

From conda (this suggests to install [pytorch nightly release](https://anaconda.org/pytorch-nightly/pytorch) instead of stable
version as dependency):

```bash
conda install ignite -c pytorch-nightly
```


<!-- ############################################################################################################### -->


# Getting Started

Few pointers to get you started:

- [Quick Start Guide: Essentials of getting a project up and running](https://pytorch.org/ignite/quickstart.html)
- [Concepts of the library: Engine, Events & Handlers, State, Metrics](https://pytorch.org/ignite/concepts.html)
- Full-featured template examples (coming soon ...)


<!-- ############################################################################################################### -->


# Documentation

- Stable API documentation and an overview of the library: https://pytorch.org/ignite/
- Development version API documentation: https://pytorch.org/ignite/master/
- [FAQ](https://pytorch.org/ignite/faq.html), 
["Questions on Github"](https://github.com/pytorch/ignite/issues?q=is%3Aissue+label%3Aquestion+) and 
["Questions on Discuss.PyTorch"](https://discuss.pytorch.org/c/ignite).
- [Project's Roadmap](https://github.com/pytorch/ignite/wiki/Roadmap)

## Additional Materials

- [8 Creators and Core Contributors Talk About Their Model Training Libraries From PyTorch Ecosystem](https://neptune.ai/blog/model-training-libraries-pytorch-ecosystem?utm_source=reddit&utm_medium=post&utm_campaign=blog-model-training-libraries-pytorch-ecosystem)
- Ignite Posters from Pytorch Developer Conferences:
    - [2019](https://drive.google.com/open?id=1bqIl-EM6GCCCoSixFZxhIbuF25F2qTZg)
    - [2018](https://drive.google.com/open?id=1_2vzBJ0KeCjGv1srojMHiJRvceSVbVR5)


<!-- ############################################################################################################### -->


# Examples

Complete list of examples can be found [here](https://pytorch.org/ignite/examples.html).

## Tutorials

-   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/TextCNN.ipynb)  [Text Classification using Convolutional Neural
    Networks](https://github.com/pytorch/ignite/blob/master/examples/notebooks/TextCNN.ipynb) 
-   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/VAE.ipynb)  [Variational Auto
    Encoders](https://github.com/pytorch/ignite/blob/master/examples/notebooks/VAE.ipynb) 
-   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/FashionMNIST.ipynb)  [Convolutional Neural Networks for Classifying Fashion-MNIST
    Dataset](https://github.com/pytorch/ignite/blob/master/examples/notebooks/FashionMNIST.ipynb)
-   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/CycleGAN_with_nvidia_apex.ipynb)  [Training Cycle-GAN on Horses to
    Zebras with Nvidia/Apex](https://github.com/pytorch/ignite/blob/master/examples/notebooks/CycleGAN_with_nvidia_apex.ipynb) - [ logs on W&B](https://app.wandb.ai/vfdev-5/ignite-cyclegan-apex) 
-   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/CycleGAN_with_torch_cuda_amp.ipynb)  [Another training Cycle-GAN on Horses to
    Zebras with Native Torch CUDA AMP](https://github.com/pytorch/ignite/blob/master/examples/notebooks/CycleGAN_with_torch_cuda_amp.ipynb) - [logs on W&B](https://app.wandb.ai/vfdev-5/ignite-cyclegan-torch-amp) 
-   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/EfficientNet_Cifar100_finetuning.ipynb)  [Finetuning EfficientNet-B0 on
    CIFAR100](https://github.com/pytorch/ignite/blob/master/examples/notebooks/EfficientNet_Cifar100_finetuning.ipynb)
-   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/Cifar10_Ax_hyperparam_tuning.ipynb)  [Hyperparameters tuning with
    Ax](https://github.com/pytorch/ignite/blob/master/examples/notebooks/Cifar10_Ax_hyperparam_tuning.ipynb) 
-   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/FastaiLRFinder_MNIST.ipynb)  [Basic example of LR finder on 
    MNIST](https://github.com/pytorch/ignite/blob/master/examples/notebooks/FastaiLRFinder_MNIST.ipynb)
-   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/Cifar100_bench_amp.ipynb)  [Benchmark mixed precision training on Cifar100: 
    torch.cuda.amp vs nvidia/apex](https://github.com/pytorch/ignite/blob/master/examples/notebooks/Cifar100_bench_amp.ipynb) 
-   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/MNIST_on_TPU.ipynb)  [MNIST training on a single 
    TPU](https://github.com/pytorch/ignite/blob/master/examples/notebooks/MNIST_on_TPU.ipynb)
-   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1E9zJrptnLJ_PKhmaP5Vhb6DTVRvyrKHx) [CIFAR10 Training on multiple TPUs](https://github.com/pytorch/ignite/tree/master/examples/contrib/cifar10)

## Reproducible Training Examples

Inspired by [torchvision/references](https://github.com/pytorch/vision/tree/master/references),
we provide several reproducible baselines for vision tasks:

-   [ImageNet](examples/references/classification/imagenet) - logs on Ignite Trains server coming soon ...
-   [Pascal VOC2012](examples/references/segmentation/pascal_voc2012) - logs on Ignite Trains server coming soon ...

Features:

-   Distributed training with mixed precision by [nvidia/apex](https://github.com/NVIDIA/apex/)
-   Experiments tracking with [MLflow](https://mlflow.org/), [Polyaxon](https://polyaxon.com/) or [TRAINS](https://github.com/allegroai/trains/) 


<!-- ############################################################################################################### -->


# Communication

- [GitHub issues](https://github.com/pytorch/ignite/issues): questions, bug reports, feature requests, etc.

- [Discuss.PyTorch](https://discuss.pytorch.org/c/ignite), category "Ignite".

- [PyTorch Slack](https://pytorch.slack.com) at #pytorch-ignite channel. [Request access](https://bit.ly/ptslack).

## User feedback

We have created a form for ["user feedback"](https://github.com/pytorch/ignite/issues/new/choose). We
appreciate any type of feedback and this is how we would like to see our
community:

-   If you like the project and want to say thanks, this the right
    place.
-   If you do not like something, please, share it with us and we can
    see how to improve it.

Thank you !


<!-- ############################################################################################################### -->


# Contributing

Please see the [contribution guidelines](https://github.com/pytorch/ignite/blob/master/CONTRIBUTING.md) for more information.

As always, PRs are welcome :)


<!-- ############################################################################################################### -->


# Projects using Ignite

## Research papers

-   [BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning](https://github.com/BlackHC/BatchBALD)
-   [A Model to Search for Synthesizable Molecules](https://github.com/john-bradshaw/molecule-chef)
-   [Localised Generative Flows](https://github.com/jrmcornish/lgf)
-   [Extracting T Cell Function and Differentiation Characteristics from the Biomedical Literature](https://github.com/hammerlab/t-cell-relation-extraction)
-   [Variational Information Distillation for Knowledge Transfer](https://github.com/amzn/xfer/tree/master/var_info_distil)
-   [XPersona: Evaluating Multilingual Personalized Chatbot](https://github.com/HLTCHKUST/Xpersona)
-   [CNN-CASS: CNN for Classification of Coronary Artery Stenosis Score in MPR Images](https://github.com/ucuapps/CoronaryArteryStenosisScoreClassification)
-   [Bridging Text and Video: A Universal Multimodal Transformer for Video-Audio Scene-Aware Dialog](https://github.com/ictnlp/DSTC8-AVSD)
-   [Adversarial Decomposition of Text Representation](https://github.com/text-machine-lab/adversarial_decomposition)
-   [Uncertainty Estimation Using a Single Deep Deterministic Neural Network](https://github.com/y0ast/deterministic-uncertainty-quantification)

## Blog articles, tutorials, books

-   [State-of-the-Art Conversational AI with Transfer Learning](https://github.com/huggingface/transfer-learning-conv-ai)
-   [Tutorial on Transfer Learning in NLP held at NAACL 2019](https://github.com/huggingface/naacl_transfer_learning_tutorial)
-   [Deep-Reinforcement-Learning-Hands-On-Second-Edition, published by Packt](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition)
-   [Once Upon a Repository: How to Write Readable, Maintainable Code with PyTorch](https://towardsdatascience.com/once-upon-a-repository-how-to-write-readable-maintainable-code-with-pytorch-951f03f6a829)
-   [The Hero Rises: Build Your Own SSD](https://allegro.ai/blog/the-hero-rises-build-your-own-ssd/)
-   [Using Optuna to Optimize PyTorch Ignite Hyperparameters](https://medium.com/optuna/using-optuna-to-optimize-pytorch-ignite-hyperparameters-b680f55ac746)

## Toolkits

-   [Project MONAI - AI Toolkit for Healthcare Imaging](https://github.com/Project-MONAI/MONAI)
-   [DeepSeismic - Deep Learning for Seismic Imaging and Interpretation](https://github.com/microsoft/seismic-deeplearning)
-   [Nussl - a flexible, object oriented Python audio source separation library](https://github.com/nussl/nussl)

## Others

-   [Implementation of "Attention is All You Need" paper](https://github.com/akurniawan/pytorch-transformer)
-   [Implementation of DropBlock: A regularization method for convolutional networks in PyTorch](https://github.com/miguelvr/dropblock)
-   [Kaggle Kuzushiji Recognition: 2nd place solution](https://github.com/lopuhin/kaggle-kuzushiji-2019)
-   [Unsupervised Data Augmentation experiments in PyTorch](https://github.com/vfdev-5/UDA-pytorch)
-   [Hyperparameters tuning with Optuna](https://github.com/pfnet/optuna/blob/master/examples/pytorch_ignite_simple.py)
-   [Logging with ChainerUI](https://chainerui.readthedocs.io/en/latest/reference/module.html#external-library-support)

See other projects at ["Used by"](https://github.com/pytorch/ignite/network/dependents?package_id=UGFja2FnZS02NzI5ODEwNA%3D%3D)

If your project implements a paper, represents other use-cases not
covered in our official tutorials, Kaggle competition's code or just
your code presents interesting results and uses Ignite. We would like to
add your project in this list, so please send a PR with brief
description of the project.


<!-- ############################################################################################################### -->

# About the team & Disclaimer

This repository is operated and maintained by volunteers in the PyTorch community in their capacities as individuals (and not as representatives of their employers). See the ["About us"](https://pytorch.org/ignite/master/about.html) page for a list of core contributors. For usage questions and issues, please see the various channels [here](https://github.com/pytorch/ignite#communication). For all other questions and inquiries, please send an email to contact@pytorch-ignite.ai.
