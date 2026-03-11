# CIFAR10 Example with Ray Tune and Ignite

This example demonstrates how to use **Ray Tune** for hyperparameter tuning with **PyTorch Ignite** for training and validation on the CIFAR10 dataset.

In this example, we show how to use _Ignite_ to train a neural network with:

- hyperparameter search using Ray Tune,
- ASHA scheduler for early stopping, see also: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.ASHAScheduler.html

## Requirements

- pytorch-ignite: `pip install pytorch-ignite`
- torch: `pip install torch`
- torchvision: `pip install torchvision`
- [Ray Tune](https://docs.ray.io/): `pip install ray[tune]`

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
| `--num_trials`     | 10       | Number of trials for Ray Tune |
| `--num_epochs`     | 10       | Number of epochs per trial    |
| `--gpus_per_trial` | 1        | GPUs per trial (0 for CPU)    |
| `--cpus_per_trial` | 4        | CPUs per trial                |
| `--data_dir`       | `./data` | Path to CIFAR10 dataset       |

### Examples

Run with custom settings:

```bash
python main.py --num_trials 5 --num_epochs 5 --gpus_per_trial 1
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

| Parameter    | Values                       |
| ------------ | ---------------------------- |
| `l1`         | [4, 8, 16, 32, 64, 128, 256] |
| `l2`         | [4, 8, 16, 32, 64, 128, 256] |
| `lr`         | [1e-4, 1e-1]                 |
| `batch_size` | [256, 512, 1024]             |

- `l1`: First fully connected layer size
- `l2`: Second fully connected layer size
- `lr`: Log-uniformly sampled learning rate in [1e-4, 1e-1] for SGD optimizer
- `batch_size`: Training batch size

## Output

If we run main.py an example output could look like:

```
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                      status         l1     l2            lr     batch_size     iter     total time (s)      loss     accuracy │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_with_ignite_ab528_00000   TERMINATED      4    128   0.000202867           1024        1            4.28721   2.31388       0.0993 │
│ train_with_ignite_ab528_00001   TERMINATED    128      4   0.0180925              256       10           20.296     1.24503       0.5711 │
│ train_with_ignite_ab528_00002   TERMINATED      8    256   0.000386848           1024        1            4.01333   2.30241       0.1003 │
│ train_with_ignite_ab528_00003   TERMINATED     16      4   0.00105969             256        1            3.59295   2.32562       0.0995 │
│ train_with_ignite_ab528_00004   TERMINATED     16    128   0.000495626           1024        2            5.8557    2.30494       0.0876 │
│ train_with_ignite_ab528_00005   TERMINATED     16    128   0.0379958              512       10           20.3416    1.20055       0.5765 │
│ train_with_ignite_ab528_00006   TERMINATED      4     32   0.00186652            1024        1            4.06322   2.30845       0.1059 │
│ train_with_ignite_ab528_00007   TERMINATED     16     16   0.00687211             512        2            5.40865   2.12892       0.2006 │
│ train_with_ignite_ab528_00008   TERMINATED      4    256   0.000143329           1024        1            3.79915   2.30698       0.0908 │
│ train_with_ignite_ab528_00009   TERMINATED     32    256   0.000630852            512        2            5.72633   2.29931       0.1221 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Best trial config: {'l1': 16, 'l2': 128, 'lr': 0.03799583796432589, 'batch_size': 512, 'device': 'cuda'}
Best trial final validation loss: 1.200553125
Best trial final validation accuracy: 0.5765
Best trial test set accuracy: 0.5688
```

We see that most trials stop earlier than the number of maximum epochs (= 10, here). This is because the ASHAScheduler performs early stopping:
after each reported epoch, it compares trial performance and prunes the worst-performing ones. So `num_epochs = 10` is the upper limit
here, most trials stop earlier.
