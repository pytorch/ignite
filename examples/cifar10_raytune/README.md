# CIFAR10 Example with Ray Tune and Ignite

This example demonstrates how to use **Ray Tune** for hyperparameter tuning with **PyTorch Ignite** for training and validation on the CIFAR10 dataset.

In this example, we show how to use _Ignite_ to train a neural network with:

- hyperparameter search using Ray Tune,
- ASHA scheduler for early stopping

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

| Parameter    | Values                       |
| ------------ | ---------------------------- |
| `l1`         | [4, 8, 16, 32, 64, 128, 256] |
| `l2`         | [4, 8, 16, 32, 64, 128, 256] |
| `lr`         | [1e-4, 1e-1]                 |
| `batch_size` | [2, 4, 8, 16]                |

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
│ train_with_ignite_f740a_00000   TERMINATED     16    256   0.000174547              8       10           166.306    1.26467       0.5487 │
│ train_with_ignite_f740a_00001   TERMINATED     64     32   0.0211235               16        1            10.6549   2.03642       0.3094 │
│ train_with_ignite_f740a_00002   TERMINATED    128     32   0.000597011              4       10           351.698    1.16781       0.6184 │
│ train_with_ignite_f740a_00003   TERMINATED     32     64   0.0898379               16        1            10.2487   2.32586       0.0985 │
│ train_with_ignite_f740a_00004   TERMINATED     32     64   0.00243182               4        2            71.8534   1.52549       0.4728 │
│ train_with_ignite_f740a_00005   TERMINATED    128     16   0.000177485              8        1            18.2198   1.96733       0.2845 │
│ train_with_ignite_f740a_00006   TERMINATED    256     16   0.00147097               8       10           165.496    1.23385       0.6033 │
│ train_with_ignite_f740a_00007   TERMINATED     16     16   0.0402147               16        1            10.5927   2.24759       0.1452 │
│ train_with_ignite_f740a_00008   TERMINATED    256      8   0.000176101              4        2            71.1053   1.51049       0.4422 │
│ train_with_ignite_f740a_00009   TERMINATED    128     32   0.000174279              4        4           139.638    1.31689       0.5304 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Best trial config: {'l1': 128, 'l2': 32, 'lr': 0.0005970108807327366, 'batch_size': 4, 'device': '...'}
Best trial final validation loss: 1.16781484375
Best trial final validation accuracy: 0.6184
Best trial test set accuracy: 0.6154
```

We see that most trials stop earlier than the number of maximum epochs (= 10, here).
