# Basic Multi-processing Distributed Data Parallel Training with Dali


## Requirements:

- [nvidia-dali](https://github.com/NVIDIA/DALI): see the [README](https://github.com/NVIDIA/DALI/blob/master/README.rst)
- [torchvision](https://github.com/pytorch/vision/): `pip install torchvision`
- [tqdm](https://github.com/tqdm/tqdm/): `pip install tqdm`


## Single node, multiple GPUs:

Run the example:
```
 python -m torch.distributed.launch --nproc_per_node=N_GPUS imagefolder_with_dali.py --root PATH/TO/DATASET --model MODEL_NAME --epochs EPOCHS --batch_size SIZE
```

## Usage

```
usage: imagefolder_with_dali.py [-h] [--root ROOT] [--model MODEL]
                                [--batch_size BATCH_SIZE]
                                [--val_batch_size VAL_BATCH_SIZE]
                                [--epochs EPOCHS] [--lr LR]
                                [--momentum MOMENTUM] [--val_ratio VAL_RATIO]
                                [--requires_grad REQUIRES_GRAD]
                                [--pretrained PRETRAINED]
                                [--local_rank LOCAL_RANK]

optional arguments:
  -h, --help            show this help message and exit
  --root ROOT           Path to the root of the dataset
  --model MODEL         Name of the model in <alexnet|densenet121|densenet161|
                        densenet169|densenet201|inception_v3|resnet101|resnet1
                        52|resnet18|resnet34|resnet50|squeezenet1_0|squeezenet
                        1_1|vgg11|vgg11_bn|vgg13|vgg13_bn|vgg16|vgg16_bn|vgg19
                        |vgg19_bn>
  --batch_size BATCH_SIZE
                        input batch size for training (default: 64)
  --val_batch_size VAL_BATCH_SIZE
                        input batch size for validation (default: 1000)
  --epochs EPOCHS       number of epochs to train (default: 10)
  --lr LR               learning rate (default: 0.01)
  --momentum MOMENTUM   SGD momentum (default: 0.5)
  --val_ratio VAL_RATIO
                        ratio of images to use for validation
  --requires_grad REQUIRES_GRAD
                        Finetune model
  --pretrained PRETRAINED
                        Use pretrained model
  --local_rank LOCAL_RANK
                        Local rank

```
