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

## Check resume training

### Single GPU

Initial training with a crash at 1000 iteration (~10 epochs)
```bash
python main.py --params="crash_iteration=1000"
# or same in deterministic mode
python main.py --params="crash_iteration=1000;deterministic=True;output_path=/tmp/output-cifar10/deterministic"
```

Resume from the latest checkpoint
```bash
python main.py --params="resume_from=/tmp/output-cifar10/XYZ-single-gpu/training_checkpoint_800.pt"
# or same in deterministic mode
python main.py --params="resume_from=/tmp/output-cifar10/deterministic/XYZ-single-gpu/training_checkpoint_800.pt;deterministic=True;output_path=/tmp/output-cifar10/deterministic" 
```

Training without crashing
```bash
python main.py
# or same in deterministic mode
python main.py --params="deterministic=True;output_path=/tmp/output-cifar10/deterministic"
```

Non-deterministic| Deterministic
---|---
![img11](assets/tb_logger_run_resume_ndet.png) | ![img12](assets/tb_logger_run_resume_det.png) 

**Note 1:** We can observe a gap on `train/batch_loss` curves between intial training and resumed training. This metric is 
computed as a running average and while resuming the training, the cumulative part is not restored.


Relative performances comparision
![img13](assets/tb_logger_det_vs_ndet.png)

**Note 2:** Please, keep in mind, that `torch.backends.cudnn.deterministic=True` and 
`torch.backends.cudnn.benchmark=False` used in [`DeterministicEngine`](https://pytorch.org/ignite/engine.html#ignite.engine.deterministic.DeterministicEngine)
have a negative single-run performance impact (see [official torch notes](https://pytorch.org/docs/stable/notes/randomness.html#cudnn)).


### Single Node, multiple GPUs

Initial training with a crash at 1000 iteration (~10 epochs)
```bash
python -u -m torch.distributed.launch --nproc_per_node=2 main.py --params="dist_backend='nccl';crash_iteration=1000"
# or same in deterministic mode
python -u -m torch.distributed.launch --nproc_per_node=2 main.py --params="dist_backend='nccl';crash_iteration=1000;deterministic=True;output_path=/tmp/output-cifar10/deterministic"
```

Resume from the latest checkpoint
```bash
python -u -m torch.distributed.launch --nproc_per_node=2 main.py --params="dist_backend='nccl';resume_from=/tmp/output-cifar10/XYZ-distributed-1nodes-2gpus/training_checkpoint_800.pt"
# or same in deterministic mode
python -u -m torch.distributed.launch --nproc_per_node=2 main.py --params="dist_backend='nccl';resume_from=/tmp/output-cifar10/deterministic/XYZ-distributed-1nodes-2gpus/training_checkpoint_800.pt;deterministic=True;output_path=/tmp/output-cifar10/deterministic" 
```

Training without crashing
```bash
python -u -m torch.distributed.launch --nproc_per_node=2 main.py --params="dist_backend='nccl'"
# or same in deterministic mode
python -u -m torch.distributed.launch --nproc_per_node=2 main.py --params="dist_backend='nccl';deterministic=True;output_path=/tmp/output-cifar10/deterministic"
```
