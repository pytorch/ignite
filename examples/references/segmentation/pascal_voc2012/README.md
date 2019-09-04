# Reproducible PASCAL VOC2012 training with Ignite

In this example, we provide script and tools to perform reproducible experiments on training neural networks on PASCAL VOC2012
dataset.

Features:

- Distributed training with mixed precision by [nvidia/apex](https://github.com/NVIDIA/apex/)

## Requirements

We use `conda` and [mlflow](https://github.com/mlflow/mlflow) to handle experiments/runs and all python dependencies. 
Please, install these tools:

- [mlflow](https://github.com/mlflow/mlflow): `pip install mlflow`
- [conda](https://conda.io/en/latest/miniconda.html)


## Dataset

To configure the path to already existing PASCAL VOC2012 dataset, please specify `DATASET_PATH` environment variable
```
export DATASET_PATH=/path/to/pascal_voc2012
```
### With SBD dataset

Optionally, user can configure the path to already existing SBD dataset, please specify `SBD_DATASET_PATH` environment variable
```
export SBD_DATASET_PATH=/path/to/sbd
# e.g. SBD_DATASET_PATH=/path/to/sbd/benchmark_RELEASE/dataset/
```

## Usage

Setup mlflow output path as 
```bash
export MLFLOW_TRACKING_URI=/path/to/output/mlruns
# e.g export MLFLOW_TRACKING_URI=$PWD/output/mlruns
```

Create once "Trainings" experiment
```
mlflow experiments create -n Trainings
```
or check existing experiments:
```
mlflow experiments list
```

### Single node with multiple GPUs

```bash
export MLFLOW_TRACKING_URI=/path/to/output/mlruns
# e.g export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/ --experiment-name=Trainings -P config_path=configs/train/baseline_resnet101.py -P num_gpus=2
```

## Training tracking

### MLflow dashboard

To visualize experiments and runs, user can start mlflow dashboard:

```bash
mlflow server --backend-store-uri /path/to/output/mlruns --default-artifact-root /path/to/output/mlruns -p 6026 -h 0.0.0.0
# e.g mlflow server --backend-store-uri $PWD/output/mlruns --default-artifact-root $PWD/output/mlruns -p 6026 -h 0.0.0.0
```

### Tensorboard dashboard

To visualize experiments and runs, user can start tensorboard:

```bash
tensorboard --logdir /path/to/output/mlruns/1
# e.g tensorboard --logdir $PWD/output/mlruns/1
```
where `/1` points to "Training" experiment. 


## Implementation details

Files tree description:
```
code
  |___ dataflow : module privides data loaders and various transformers
  |___ scripts : executable training and inference scripts
  |___ tools : other helper modules

configs
  |___ train : training python configuration files
  |___ inference : inference python configuration files
  
experiments : MLflow related files

notebooks : jupyter notebooks to check specific parts from code modules 
```

### Experiments

- [conda.yaml](experiments/conda.yaml): defines all python dependencies necessary for our experimentations


- [MLproject](experiments/MLproject): defines types of experiments we would like to perform by "entry points":
  - main : starts single-node multi-GPU training script

When we execute 
```
mlflow run experiments/ --experiment-name=Trainings -P config_path=configs/train/baseline_resnet101.py -P num_gpus=2
```
it executes `main` entry point from [MLproject](experiments/MLproject) and runs provided command.

### Code and configs

#### [py_config_runner](https://github.com/vfdev-5/py_config_runner)

We use [py_config_runner](https://github.com/vfdev-5/py_config_runner) package to execute python scripts with python configuration files.






