# Experiments tracking with Polyaxon

User can run ImageNet training using [Polyaxon experiments tracking system](https://polyaxon.com/).

## Requirements

In this case we assume, user has [Polyaxon](https://polyaxon.com/) installed on a machine/cluster/cloud and can schedule experiments with `polyaxon-cli`.

## Usage

### Setup Polyaxon project

Create project on the cluster

```bash
polyaxon project create --name=imagenet --description="Classification on ImageNet"
```

Initialize local project

```bash
polyaxon init imagenet
```

Please rename and modify `experiments/plx/xp_training.yml.tmpl` to `experiments/plx/xp_training.yml`
to adapt to your cluster configuration.

#### Download ImageNet dataset

Since 10/2019, we need to register an account in order to download the dataset.
To download the dataset, use the following form : http://www.image-net.org/download.php

### Training on single node with single or multiple GPU

For optimal devices usage, please, make sure to adapt training data loader batch size to your infrastructure.
By default, batch size is 64 per process. Please, adapt `xp_training.yml` to your cluster configuration and run it, for example, as

```bash
polyaxon run -u -f experiments/plx/xp_training.yml --name="baseline_resnet50" --tags=train,resnet50
```

## Training tracking

Please, see Polyaxon dashboard usage at https://docs.polyaxon.com/

## Implementation details

Files tree description:

```
code
configs
experiments/plx : Polyaxon related files
notebooks
```

### Experiments

File [xp_training.yml.tmpl](experiments/mlflow/xp_training.yml.tmpl) defines all configurations and dependencies
necessary for our experimentations. Part `run.cmd` starts single-node multi-GPU training script.
