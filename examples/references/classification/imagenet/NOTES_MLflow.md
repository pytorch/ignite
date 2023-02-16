# Experiments tracking with MLflow

User can run ImageNet training using MLflow experiments tracking system on the local machine.

## Requirements

We use `conda` and [MLflow](https://github.com/mlflow/mlflow) to
handle experiments/runs and all python dependencies.
Please, install these tools:

- [MLflow](https://github.com/mlflow/mlflow): `pip install mlflow`
- [conda](https://conda.io/en/latest/miniconda.html)

We need to also install Nvidia/APEX and libraries for opencv. APEX is automatically installed on the first run.
Manually, all can be installed with the following commands.
**Important**, please, check the content of `experiments/setup_opencv.sh` before running.

```bash
sh experiments/setup_apex.sh

sh experiments/setup_opencv.sh
```

## Usage

### Download ImageNet-1k dataset

Since 10/2019, we need to register an account in order to download the dataset.
To download the dataset, use the following form : http://www.image-net.org/download.php

### Setup dataset path

To configure the path to already existing ImageNet dataset, please specify `DATASET_PATH` environment variable

```bash
export DATASET_PATH=/path/to/imagenet
# export DATASET_PATH=$PWD/input/imagenet
```

### MLflow setup

Setup mlflow output path as a local storage (option with remote storage is not supported):

```bash
export MLFLOW_TRACKING_URI=/path/to/output/mlruns
# e.g export MLFLOW_TRACKING_URI=$PWD/output/mlruns
```

Create once "Trainings" experiment

```bash
mlflow experiments create -n Trainings
```

or check existing experiments:

```bash
mlflow experiments list
```

### Training on single node with single GPU

Please, make sure to adapt training data loader batch size to your GPU type. By default, batch size is 64.

```bash
export MLFLOW_TRACKING_URI=/path/to/output/mlruns
# e.g export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/mlflow --experiment-name=Trainings -P config_path=configs/train/baseline_r50.py -P num_gpus=1
```

### Training on single node with multiple GPUs

For optimal devices usage, please, make sure to adapt training data loader batch size to your infrastructure.
By default, batch size is 64 per process.

```bash
export MLFLOW_TRACKING_URI=/path/to/output/mlruns
# e.g export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/mlflow --experiment-name=Trainings -P config_path=configs/train/baseline_r50.py -P num_gpus=2
```

## Training tracking

### MLflow dashboard

To visualize experiments and runs, user can start mlflow dashboard:

```bash
mlflow server --backend-store-uri /path/to/output/mlruns --default-ainfrastructure/path/to/output/mlruns -p 6026 -h 0.0.0.0
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
configs
experiments/mlflow : MLflow related files
notebooks
```

### Experiments

- [conda.yaml](experiments/mlflow/conda.yaml): defines all python dependencies necessary for our experimentations
- [MLproject](experiments/mlflow/MLproject): defines types of experiments we would like to perform by "entry points":
  - main : starts single-node multi-GPU training script

When we execute

```bash
mlflow run experiments/mlflow --experiment-name=Trainings -P config_path=configs/train/baseline_r50.py -P num_gpus=2
```

it executes `main` entry point from [MLproject](experiments/mlflow/MLproject) and runs provided command.
