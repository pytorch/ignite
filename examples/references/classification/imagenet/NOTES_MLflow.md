# Experiments tracking with MLflow
  
## Requirements

We use `conda` and [MLflow](https://github.com/mlflow/mlflow) to 
handle experiments/runs and all python dependencies. 
Please, install these tools:

- [MLflow](https://github.com/mlflow/mlflow): `pip install mlflow`
- [conda](https://conda.io/en/latest/miniconda.html)


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
 
Setup mlflow output path as 
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

### Single node with multiple GPUs

```bash
export MLFLOW_TRACKING_URI=/path/to/output/mlruns
# e.g export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/mlflow --experiment-name=Trainings -P config_path=configs/train/baseline_r50.py -P num_gpus=2
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
