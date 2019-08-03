# CIFAR10 Example with Ignite

In this example, we show how to use *Ignite* to train a neural network on 1 or more GPUs, save the best model weights, 
log learning rate, training/validation metrics.

Configurations:
[x] single GPU
[x] multi GPUs on a single node
[ ] multi GPUs on multiple nodes

## Requirements:

- [torchvision](https://github.com/pytorch/vision/): `pip install torchvision`
- [tqdm](https://github.com/tqdm/tqdm/): `pip install tqdm`
- [tensorboardx](https://github.com/lanpa/tensorboard-pytorch): `pip install tensorboardX`

## Usage:

![tb](assets/tb_logger.png)

Run the example on a single GPU (script will not run without a GPU):
```bash
python main.py
```

### Distributed training

#### Single node, multiple GPUs
```bash
python -u -m torch.distributed.launch --nproc_per_node=2 main.py --params="batch_size=512;dist_backend='nccl'"
```

#### Multiple nodes, multiple GPUs

```bash
todo
```

## Acknowledgements

In this repository we are using the code from 
- [cifar10-fast repository](https://github.com/davidcpage/cifar10-fast)

Thanks to the authors for sharing their code!
