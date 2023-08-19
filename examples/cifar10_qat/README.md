# Example of Quantization Aware Training (QAT) with Ignite on CIFAR10

Model's implementation is based on https://discuss.pytorch.org/t/evaluator-returns-nan/107972/3

In this example, we show how to use _Ignite_ to train a neural network:

- on 1 or more GPUs
- compute training/validation metrics
- log learning rate, metrics etc
- save the best model weights

Configurations:

- [x] single GPU
- [x] multi GPUs on a single node

## Requirements:

- pytorch-ignite: `pip install pytorch-ignite`
- [torchvision](https://github.com/pytorch/vision/): `pip install torchvision`
- [tqdm](https://github.com/tqdm/tqdm/): `pip install tqdm`
- [tensorboardx](https://github.com/lanpa/tensorboard-pytorch): `pip install tensorboardX`
- [python-fire](https://github.com/google/python-fire): `pip install fire`
- [brevitas](https://github.com/Xilinx/brevitas): `pip install git+https://github.com/Xilinx/brevitas.git`

## Usage:

We can train, for example, ResNet-18 with 8 bit weights and activations.

Run the example on a single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py run --model="resnet18_QAT_8b"
```

Note: torch DataParallel is not working (v1.7.1) with QAT.

For details on accepted arguments:

```bash
python main.py run -- --help
```

If user would like to provide already downloaded dataset, the path can be setup in parameters as

```bash
--data_path="/path/to/cifar10/"
```

Other available models can be found [here](utils.py):

- resnet18_QAT_8b - ResNet-18 with 8 bit weights and activations
- resnet18_QAT_6b - ResNet-18 with 6 bit weights and activations
- resnet18_QAT_5b - ResNet-18 with 5 bit weights and activations
- resnet18_QAT_4b - ResNet-18 with 4 bit weights and activations
- torchvision models

### Distributed training

#### Single node, multiple GPUs

Let's start training on a single node with 2 gpus:

```bash
# using torch.distributed.launch
python -u -m torch.distributed.launch --nproc_per_node=2 --use_env main.py run --backend="nccl" --model="resnet18_QAT_8b"
```

##### Using [Horovod](https://horovod.readthedocs.io/en/latest/index.html) as distributed backend

Please, make sure to have Horovod installed before running.

Let's start training on a single node with 2 gpus:

```bash
# horovodrun
horovodrun -np=2 python -u main.py run --backend="horovod" --model="resnet18_QAT_8b"
```

or

```bash
# using function spawn inside the code
python -u main.py run --backend="horovod" --nproc_per_node=2 --model="resnet18_QAT_8b"
```

### Online logs

On TensorBoard.dev: https://tensorboard.dev/experiment/Kp9Wod3XR36Sg2I1gAh1cA/
