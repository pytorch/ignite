# Reproducible ImageNet training with Ignite

In this example, we provide script and tools to perform reproducible experiments on training neural networks on ImageNet
dataset.

Features:

- Distributed training with native automatic mixed precision
- Experiments tracking with [ClearML](https://github.com/allegroai/clearml)

| Model     | Training Top-1 Accuracy | Training Top-5 Accuracy | Test Top-1 Accuracy | Test Top-5 Accuracy |
| --------- | ----------------------- | ----------------------- | ------------------- | ------------------- |
| ResNet-50 | 78%                     | 92%                     | 77%                 | 94%                 |

| Experiment     | Model | Training Top-1 Accuracy | Training Top-5 Accuracy | Test Top-1 Accuracy | Test Top-5 Accuracy | ClearML Link |
| -------------- | ----- | ----------------------- | ----------------------- | ------------------- | ------------------- | ------------ |
| configs/???.py |

## Setup

```
pip install -r requirements.txt
```

### Docker

For docker users, you can use the following images to run the example:

```bash
docker pull pytorchignite/vision:latest
```

and install other requirements as suggested above

## Usage

Please, export the `DATASET_PATH` environment variable for the ImageNet dataset.

```bash
export DATASET_PATH=/path/to/imagenet
# e.g. export DATASET_PATH=/data/ where "train", "val", "meta.bin" are located
```

### Training

#### Single GPU

- Adjust batch size for your GPU type in the configuration file: `configs/baseline_resnet50.py` or `configs/baseline_resnet50.py`

Run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python -u main.py training configs/baseline_resnet50.py
```

#### Multiple GPUs

- Adjust total batch size for your GPUs in the configuration file: `configs/baseline_resnet50.py` or `configs/baseline_resnet50.py`

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 main.py training configs/baseline_resnet50.py
```

## Acknowledgements

Trainings were done using credits provided by [trainml.ai](trainml.ai) platform.
