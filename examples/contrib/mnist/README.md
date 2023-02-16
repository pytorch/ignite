# Basic MNIST Example with Ignite and `ignite.contrib` module

ported from [pytorch-examples](https://github.com/pytorch/examples/tree/master/mnist)

Basic neural network training with Ignite and various built-in loggers from `ignite.contrib`:

- TQDM progress bar
- Tensorboard
- Visdom

### Usage:

#### Requirements:

- [torchvision](https://github.com/pytorch/vision/): `pip install torchvision`
- [tqdm](https://github.com/tqdm/tqdm/): `pip install tqdm`

#### Logging with TQDM progress bar

Run the example:

```
python mnist_with_tqdm_logger.py
```

### Logging with TensorboardX or `torch.utils.tensorboard`

Example with training and validation monitoring using Tensorboard.

#### Requirements:

- [torchvision](https://github.com/pytorch/vision/): `pip install torchvision`
- Optionally [TensorboardX](https://github.com/lanpa/tensorboard-pytorch): `pip install tensorboardX`
- Tensorboard: `pip install tensorboard`

Optionally, user can install `pynvml` package on Python 3 and log GPU information: used memory, utilization.

#### Usage:

Run the example:

```bash
python mnist_with_tensorboard_logger.py --log_dir=/tmp/tensorboard_logs
```

Start tensorboard:

```bash
tensorboard --logdir=/tmp/tensorboard_logs/
```

### Logging with Visdom

Example with training and validation monitoring using Visdom

#### Requirements:

- [torchvision](https://github.com/pytorch/vision/): `pip install torchvision`
- [Visdom](https://github.com/facebookresearch/visdom): `pip install visdom`

#### Usage:

Run the example:

```bash
python mnist_with_visdom_logger.py
```
