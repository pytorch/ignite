# Reproducible PASCAL VOC2012 training with PyTorch-Ignite

In this example, we provide script and tools to perform reproducible experiments on training neural networks on PASCAL VOC2012
dataset.

Features:

- Distributed training with native automatic mixed precision
- Experiments tracking with [ClearML](https://github.com/allegroai/clearml)

| Experiment                              | Model               | Dataset  | Val Avg IoU | ClearML Link                                                                                                                         |
| --------------------------------------- | ------------------- | -------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| configs/baseline_dplv3_resnet101.py     | DeepLabV3 Resnet101 | VOC Only | 0.659161    | [link](https://app.clear.ml/projects/0e9a3a92d3134283b7d5572d516d60c5/experiments/a7254f084a9e47ca9380dfd739f89520/output/execution) |
| configs/baseline_dplv3_resnet101_sbd.py | DeepLabV3 Resnet101 | VOC+SBD  | 0.6853087   | [link](https://app.clear.ml/projects/0e9a3a92d3134283b7d5572d516d60c5/experiments/dc4cee3377a74d19bc2d0e0e4d638c1f/output/execution) |

## Setup

```
pip install -r requirements.txt
```

### Docker

For docker users, you can use the following images to run the example:

```bash
docker pull pytorchignite/vision:latest
```

or

```bash
docker pull pytorchignite/hvd-vision:latest
```

and install other requirements as suggested above

### Using Horovod as distributed framework

We do not add `horovod` as a requirement into `requirements.txt`. Please, install it manually following the official guides or
use `pytorchignite/hvd-vision:latest` docker image.

### (Optional) Download Pascal VOC2012 and SDB datasets

Download and extract the datasets:

```bash
python main.py download /path/to/datasets
```

This script will download and extract the following datasets into `/path/to/datasets`

- The [Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) dataset
- Optionally, the [SBD](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz) evaluation dataset

## Usage

Please, export the `DATASET_PATH` environment variable for the Pascal VOC2012 dataset.

```bash
export DATASET_PATH=/path/to/pascal_voc2012
# e.g. export DATASET_PATH=/data/ where VOCdevkit is located
```

Optionally, if using SBD dataset, export the `SBD_DATASET_PATH` environment variable:

```bash
export SBD_DATASET_PATH=/path/to/SBD/
# e.g. export SBD_DATASET_PATH=/data/SBD/  where "cls  img  inst  train.txt  train_noval.txt  val.txt" are located
```

### Training

#### Single GPU

- Adjust batch size for your GPU type in the configuration file: `configs/baseline_dplv3_resnet101_sbd.py` or `configs/baseline_dplv3_resnet101.py`

Run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python -u main.py training configs/baseline_dplv3_resnet101_sbd.py
# or without SBD
# CUDA_VISIBLE_DEVICES=0 python -u main.py training configs/baseline_dplv3_resnet101.py
```

#### Multiple GPUs

- Adjust total batch size for your GPUs in the configuration file: `configs/baseline_dplv3_resnet101_sbd.py` or `configs/baseline_dplv3_resnet101.py`

```bash
torchrun --nproc_per_node=2 main.py training configs/baseline_dplv3_resnet101_sbd.py
# or without SBD
# torchrun --nproc_per_node=2 main.py training configs/baseline_dplv3_resnet101.py
```

#### Using Horovod as distributed framework

- Adjust total batch size for your GPUs in the configuration file: `configs/baseline_dplv3_resnet101_sbd.py` or `configs/baseline_dplv3_resnet101.py`

```bash
horovodrun -np=2 python -u main.py training configs/baseline_dplv3_resnet101_sbd.py --backend="horovod"
# or without SBD
# horovodrun -np=2 python -u main.py training configs/baseline_dplv3_resnet101.py --backend="horovod"
```

### Evaluation

#### Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python -u main.py eval configs/eval_baseline_dplv3_resnet101_sbd.py
```

#### Multiple GPUs

```bash
torchrun --nproc_per_node=2 main.py eval configs/eval_baseline_dplv3_resnet101_sbd.py
```

#### Using Horovod as distributed framework

```bash
horovodrun -np=2 python -u main.py eval configs/eval_baseline_dplv3_resnet101_sbd.py --backend="horovod"
```

## Acknowledgements

Trainings were done using credits provided by AWS for open-source development via NumFOCUS
and using [trainml.ai](trainml.ai) platform.
