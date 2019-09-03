# Reproducible PASCAL VOC2012 training with Ignite

In this example, we provide script and tools to perform reproducible experiments on training neural networks on PASCAL VOC2012
dataset.

Features:

- Distributed training with mixed precision by [nvidia/apex](https://github.com/NVIDIA/apex/)

## Requirements

We use `conda` and [mlflow](https://github.com/mlflow/mlflow) to handle all python dependencies. 
Please, install these tools:

- [mlflow](https://github.com/mlflow/mlflow): `pip install mlflow`
- [conda](https://conda.io/en/latest/miniconda.html)


## Dataset

To configure the path to already existing PASCAL VOC2012 dataset, please specify `DATASET_PATH` environment variable
```
export DATASET_PATH=/path/to/pascal_voc2012
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
mlflow server --backend-store-uri /path/to/output/mlruns --default-artifact-root /path/to/output/mlruns -p 6006 -h 0.0.0.0
# e.g mlflow server --backend-store-uri $PWD/output/mlruns --default-artifact-root $PWD/output/mlruns -p 6006 -h 0.0.0.0
```


