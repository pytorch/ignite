# Experiments tracking with MLflow

User can run Pascal VOC training using MLflow experiments tracking system on the local machine.

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

### Download Pascal VOC 2012 and SBD

```bash
export MLFLOW_TRACKING_URI=/path/to/output/mlruns
# e.g export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/mlflow -e download -P output_path=/path/where/download/
# e.g mlflow run experiments/mlflow -e download -P output_path=$PWD/input
```

### Setup dataset path

To configure the path to already existing PASCAL VOC2012 dataset, please specify `DATASET_PATH` environment variable
```bash
export DATASET_PATH=/path/to/pascal_voc2012
# e.g. export DATASET_PATH=$PWD/input/ where VOCdevkit is located
```

#### With SBD dataset

Optionally, user can configure the path to already existing SBD dataset, please specify `SBD_DATASET_PATH` environment variable
```bash
export SBD_DATASET_PATH=/path/to/SBD/benchmark_RELEASE/dataset/
# e.g. export SBD_DATASET_PATH=/data/SBD/benchmark_RELEASE/dataset/  where "cls  img  inst  train.txt  train_noval.txt  val.txt" are located
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

Please, make sure to adapt training data loader batch size to your GPU type. For example, a single GPU with 11GB can have a batch size of 8-9.

```bash
export MLFLOW_TRACKING_URI=/path/to/output/mlruns
# e.g export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/mlflow --experiment-name=Trainings -P config_path=configs/train/baseline_resnet101.py -P num_gpus=1
```

### Training on single node with multiple GPUs

For optimal devices usage, please, make sure to adapt training data loader batch size to your infrastructure. 
For example, a single GPU with 11GB can have a batch size of 8-9, thus, on N devices, we can set it as `N * 9`.

```bash
export MLFLOW_TRACKING_URI=/path/to/output/mlruns
# e.g export MLFLOW_TRACKING_URI=$PWD/output/mlruns
mlflow run experiments/mlflow --experiment-name=Trainings -P config_path=configs/train/baseline_resnet101.py -P num_gpus=2
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
mlflow run experiments/mlflow --experiment-name=Trainings -P config_path=configs/train/baseline_resnet101.py -P num_gpus=2
```
it executes `main` entry point from [MLproject](experiments/mlflow/MLproject) and runs provided command.
