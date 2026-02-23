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
- `lr`: Log-uniformly sampled learning rate in [1e-4, 1e-1] for SGD optimizer
- `batch_size`: Training batch size

## Output

```
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                      status         l1     l2            lr     batch_size     iter     total time (s)      loss     accuracy │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_with_ignite_aac0d_00000   TERMINATED      4      8   0.0428552               16        5            46.1855   2.30856       0.1008 │
│ train_with_ignite_aac0d_00001   TERMINATED     16      8   0.0212936               16        5            45.457    2.30575       0.0982 │
│ train_with_ignite_aac0d_00002   TERMINATED    128     16   0.0284869                8        1            19.3755   2.31148       0.0961 │
│ train_with_ignite_aac0d_00003   TERMINATED     16     16   0.0313472                2        1            67.4649   2.32548       0.1007 │
│ train_with_ignite_aac0d_00004   TERMINATED     64      8   0.000166433             16        5            46.9896   1.63962       0.3944 │
│ train_with_ignite_aac0d_00005   TERMINATED    256    256   0.000844781             16        5            45.4557   1.24878       0.5645 │
│ train_with_ignite_aac0d_00006   TERMINATED    128     64   0.0023218               16        5            47.2906   1.18214       0.5909 │
│ train_with_ignite_aac0d_00007   TERMINATED    256    256   0.000197742              8        5            83.0138   1.39471       0.4984 │
│ train_with_ignite_aac0d_00008   TERMINATED     32    256   0.028539                 2        1            68.1283   2.32266       0.0988 │
│ train_with_ignite_aac0d_00009   TERMINATED      8    256   0.000282717              2        5           334.082    1.20853       0.5667 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Best trial config: {'l1': 128, 'l2': 64, 'lr': 0.0023218034489587073, 'batch_size': 16, 'device': '...'}
Best trial final validation loss: 1.18213955078125
Best trial final validation accuracy: 0.5909
```
