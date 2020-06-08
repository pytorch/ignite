# CIFAR10 Example with Ignite

In this example, we show how to use *Ignite* to train a neural network:
- on 1 or more GPUs or TPUs
- compute training/validation metrics
- log learning rate, metrics etc
- save the best model weights

Configurations:

* [x] single GPU
* [x] multi GPUs on a single node
* [x] multi GPUs on multiple nodes
* [x] TPUs on Colab

## Requirements:

- [torchvision](https://github.com/pytorch/vision/): `pip install torchvision`
- [tqdm](https://github.com/tqdm/tqdm/): `pip install tqdm`
- [tensorboardx](https://github.com/lanpa/tensorboard-pytorch): `pip install tensorboardX`
- [python-fire](https://github.com/google/python-fire): `pip install fire`

## Usage:

Run the example on a single GPU:
```bash
python main.py run
```

For details on accepted arguments:
```bash
python main.py run -- --help
```

If user would like to provide already downloaded dataset, the path can be setup in parameters as
```bash
--data_path="/path/to/cifar10/"
```


### Distributed training

#### Single node, multiple GPUs

Let's start training on a single node with 2 gpus:
```bash
# using torch.distributed.launch
python -u -m torch.distributed.launch --nproc_per_node=2 --use_env main.py run --backend="nccl"
```
or 
```bash
# using function spawn inside the code
python -u main.py run --backend="nccl" --num_procs_per_node=2
```

If user would like to provide already downloaded dataset, the path can be setup in parameters as
```bash
--data_path="/path/to/cifar10/"
```

#### Colab, on 8 TPUs

```bash
python -u main.py run --backend="xla-tpu" --num_procs_per_node=8 --num_workers=8 --log_every_iters=0
```


#### Multiple nodes, multiple GPUs

Let's start training on two nodes with 2 gpus each. We assuming that master node can be connected as `master`, e.g. `ping master`.

1) Execute on master node
```bash
python -u -m torch.distributed.launch \
    --nnodes=2 \
    --nproc_per_node=2 \
    --node_rank=0 \
    --master_addr=master --master_port=2222 --use_env \
    main.py run --backend="nccl"
```

2) Execute on worker node
```bash
python -u -m torch.distributed.launch \
    --nnodes=2 \
    --nproc_per_node=2 \
    --node_rank=1 \
    --master_addr=master --master_port=2222 \
    main.py run --backend="nccl"
```
