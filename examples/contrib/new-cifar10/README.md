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

Run the example on a single GPU (script will not run without a GPU):
```bash
python main.py
```

If user would like to provide already downloaded dataset, the path can be setup in parameters as
```bash
--params="data_path=/path/to/cifar10/"
```


### Distributed training

#### Single node, multiple GPUs

Let's start training on a single node with 2 gpus:
```bash
python -u -m torch.distributed.launch --nproc_per_node=2 main.py --params="dist_backend='nccl'"
```

If user would like to provide already downloaded dataset, the path can be setup in parameters as
```bash
--params="data_path=/path/to/cifar10/;batch_size=512"
```

![tb1](assets/tb_logger.png)


#### Multiple nodes, multiple GPUs

Let's start training on two nodes with 2 gpus each. We assuming that master node can be connected as `master`, e.g. `ping master`.

1) Execute on master node
```bash
python -u -m torch.distributed.launch \
    --nnodes=2 \
    --nproc_per_node=2 \
    --node_rank=0 \
    --master_addr=master --master_port=2222 \
    main.py --params="dist_backend='nccl'"
```

2) Execute on worker node
```bash
python -u -m torch.distributed.launch \
    --nnodes=2 \
    --nproc_per_node=2 \
    --node_rank=1 \
    --master_addr=master --master_port=2222 \
    main.py --params="dist_backend='nccl'"
```

![tb2](assets/tb_logger_gcp.png)

## Reproduce trainings

- To reproduce trainings with [Polyaxon](https://polyaxon.com/), please see [plx_configs/README.md](plx_configs/README.md)
- To reproduce trainings on [GCP AI platform](https://cloud.google.com/ml-engine/docs/), please see [gcp_ai_platform](gcp_ai_platform/README.md).

## Acknowledgements

In this repository we are using the code from 
- [cifar10-fast repository](https://github.com/davidcpage/cifar10-fast)

Thanks to the authors for sharing their code!


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

<!---
Non-deterministic| Deterministic
---|---
![img21](assets/tb_logger_2x_run_resume_ndet.png) | ![img22](assets/tb_logger_2x_run_resume_det.png) 

Relative performances comparision
![img23](assets/tb_logger_2x_det_vs_ndet.png)

![tbresume](assets/tb_logger_resume1.png)

- Orange curves represent the training with a crash at the iteration 1000
- Blue curves show resumed training from the last checkpoint (iteration 800)
- Red curves display complete training without crashing  


**Note 2:** As we are resuming the training from an iteration between epochs, even if Ignite's engine handles the dataflow by
correctly providing data samples for the resumed iteration, random data augmentations are not synchronized. This causes a gap  
in validation curves (`train/loss`, `train/accuracy` etc.) at the begining of training resuming. 
  
![tb-resume](assets/tb_logger_resume2.png)

-->

