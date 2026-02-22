# CIFAR10 Example with Ray Tune and Ignite

This example demonstrates how to use **Ray Tune** for hyperparameter tuning with **PyTorch Ignite** for training and validation on the CIFAR10 dataset.

In this example, we show how to use _Ignite_ to train a neural network with:

- hyperparameter search using Ray Tune,
- ASHA scheduler for early stopping
- multi-GPU support

## Requirements

- pytorch-ignite: `pip install pytorch-ignite`
- torchvision: `pip install torchvision`
- [Ray Tune](https://docs.ray.io/): `pip install ray[tune]`
- ipywidgets: `pip install ipywidgets`

Alternatively, please see `requirements.txt`

```bash
pip install -r requirements.txt
```

## Usage

Run with default settings (10 trials, 10 epochs):

```bash
python main.py
```

### Command-line arguments

| Argument           | Default  | Description                   |
| ------------------ | -------- | ----------------------------- |
| `--num_samples`    | 10       | Number of trials for Ray Tune |
| `--num_epochs`     | 10       | Number of epochs per trial    |
| `--gpus_per_trial` | 0        | GPUs per trial (0 for CPU)    |
| `--cpus_per_trial` | 2        | CPUs per trial                |
| `--data_dir`       | `./data` | Path to CIFAR10 dataset       |

### Examples

Run with custom settings:

```bash
python main.py --num_samples 5 --num_epochs 5 --gpus_per_trial 1
```

Use a different data directory:

```bash
python main.py --data_dir /path/to/cifar10/
```

For more details on accepted arguments:

```bash
python main.py --help
```

## Search Space

The example searches over the following hyperparameters:

| Parameter    | Values                             |
| ------------ | ---------------------------------- |
| `l1`         | [1, 2, 4, 8, 16, 32, 64, 128, 256] |
| `l2`         | [1, 2, 4, 8, 16, 32, 64, 128, 256] |
| `lr`         | [1e-4, 1e-1]                       |
| `batch_size` | [2, 4, 8, 16]                      |

- `l1`: First fully connected layer size
- `l2`: Second fully connected layer size
- `lr`: Learning rate for SGD optimizer
- `batch_size`: Training batch size

## Output

```
Best trial config: {'l1': 128, 'l2': 64, 'lr': 0.0012, 'batch_size': 8, 'device': 'cuda'}
Best trial final validation loss: 1.2345
Best trial final validation accuracy: 0.5842
Best trial test set accuracy: 0.5842
```
