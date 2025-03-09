# Siamese Network example on CIFAR10 dataset

This example is inspired from [pytorch/examples/siamese_network](https://github.com/pytorch/examples/tree/main/siamese_network). It illustrates the implementation of Siamese Network for checking image similarity in CIFAR10 dataset.

## Usage:

```
pip install -r requirements.txt
python siamese_network.py [-h] [--batch-size BATCHSIZE] [--test-batch-size TESTBATCHSIZE] [--epochs EPOCHS]
                          [--lr LEARNINGRATE] [--gamma GAMMA] [--no-cuda][--no-mps] [--dry-run]
                          [--seed SEED] [--log-interval LOGINTERVAL] [--save-model] [--num-workers NUMWORKERS]

optional arguments:
    -h, --help          shows usage and exits
    --batch-size        sets training batch size
    --test-batch-size   sets testing batch size
    --epochs            sets number of training epochs
    --lr                sets learning rate
    --gamma             sets gamma parameter for LR Scheduler
    --no-cuda           disables CUDA training
    --no-mps            disables macOS GPU training
    --dry-run           runs model over a single pass
    --seed              sets random seed
    --log-interval      sets number of epochs before logging results
    --save-model        saves current model
    --num-workers       sets number of processes generating parallel batches
```

## Example Usage:

```
python siamese_network.py --batch-size 64 --test-batch-size 256 --epochs 14 --lr 0.95 --gamma 0.97 --num-workers 5
```
