# Reproducible PASCAL VOC2012 training with Ignite

In this example, we provide script and tools to perform reproducible experiments on training neural networks on PASCAL VOC2012
dataset.

Features:
- Distributed training with mixed precision by [nvidia/apex](https://github.com/NVIDIA/apex/)
- Experiments tracking with [MLflow](https://mlflow.org/) or [Polyaxon](https://polyaxon.com/)

![tb_dashboard](assets/tb_dashboard.png)

There are two possible options: 1) Experiments tracking with MLflow or 2) Experiments tracking with Polyaxon. 
Experiments tracking with MLflow is more suitable for a local machine with GPUs. For experiments tracking with Polyaxon
user needs to have Polyaxon installed on a machine/cluster/cloud and can schedule experiments with `polyaxon-cli`.
User can choose one option and skip the descriptions of another option.

- Notes for [experiments tracking with MLflow](NOTES_MLflow.md)
- Notes for [experiments tracking with Polyaxon](NOTES_Polyaxon.md)

## Implementation details

Files tree description:
```
code
  |___ dataflow : module privides data loaders and various transformers
  |___ scripts : executable training and inference scripts
  |___ utils : other helper modules

configs
  |___ train : training python configuration files
  |___ inference : inference python configuration files
  
experiments 
  |___ mlflow : MLflow related files
  |___ plx : Polyaxon related files
 
notebooks : jupyter notebooks to check specific parts from code modules 
```

## Code and configs

### [py_config_runner](https://github.com/vfdev-5/py_config_runner)

We use [py_config_runner](https://github.com/vfdev-5/py_config_runner) package to execute python scripts with python configuration files.

### Training scripts

Training scripts are located [code/scripts](code/scripts/) and contains  

- `mlflow_training.py`, training script with MLflow experiments tracking
- `plx_training.py`, training script with Polyaxon experiments tracking
- `gen_training.py`, common training code used by above files
 
Training scripts contain `run` method required by [py_config_runner](https://github.com/vfdev-5/py_config_runner) to 
run a script with a configuration. Training logic is setup inside `training` method and configures a disrtibuted trainer, 
2 evaluators and various logging handlers to tensorboard, mlflow/polyaxon logger and tqdm.


### Configurations

- [baseline_resnet101.py](configs/train/baseline_resnet101.py) : trains DeeplabV3-ResNet101 on Pascal VOC2012 dataset only
- [baseline_resnet101_sbd.py](configs/train/baseline_resnet101_sbd.py) : trains DeeplabV3-ResNet101 on Pascal VOC2012 dataset with SBD

