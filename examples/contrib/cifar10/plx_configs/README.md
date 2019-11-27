# Distributed training on CIFAR10 with Polyaxon

In this folder, user can find templates to run the trainings with [Polyaxon](https://polyaxon.com/).

To use the templates user needs to have
- deployed Polyaxon on a Kubernetes cluster, see [Setup and installation](https://docs.polyaxon.com/setup/)
- [Polyaxon CLI](https://docs.polyaxon.com/setup/cli/)

## Setup

- Create project on the cluster
```bash
polyaxon project create --name=cifar10-ignite-distrib --description="Distributed training on CIFAR10 with Ignite"
```

- Initialize local project from `cifar10` folder
```bash
polyaxon init cifar10-ignite-distrib
``` 

### Adapt configurations

Before running the trainings, user needs to adapt the configurations to its own Polyaxon setup.
Please, copy `xp_training_*.yaml.tmpl` to `xp_training_*.yaml` and modify new file according to your Polyaxon setup. 

## Start training

Run a training from `cifar10` folder, for example (assuming adapted configuration file `xp_training_2n_4gpus.yaml`): 
```bash
polyaxon run -u -f plx_configs/xp_training_2n_4gpus.yaml --name="distrib-training-cifar10-2n-4gpus"
```
