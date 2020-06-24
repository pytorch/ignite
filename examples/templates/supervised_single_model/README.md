# Template example for Supervised Single-Model Training

Suitable for Vision/NLP training or finetuning tasks. For example, image classification, detection or segmentation.

## Requirements

```bash
pip install torch torchvision pytorch-ignite
pip install tensorboard
```

## How to use

- Copy files to your project
- Adapt the code to your problem
  - dataflow
  - model/optimizer/criterion 
  - training step
  - metrics ([list](https://pytorch.org/ignite/metrics.html#complete-list-of-metrics) of available ignite metrics)
  - etc

All templates are optionally structured into several files:

- `main.py` - main entry-point to run training script
- `trainer.py` - helper module to setup ignite trainer
- `utils.py` - helper module to setup dataflow, model, optimizer, criterion, lr scheduler etc. 


### Training on single or multiple GPU(s) 

Current code uses fake data and can be executed in the following configurations: 

```bash
# Single process
python main.py
```
or
```bash
# Multiple processes
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py
```

### Some other best practices

- Replace configuration structure (currently, defined by simple dictionary) by
  - `OmegaDict` object and yaml files managed by [hydra](https://github.com/facebookresearch/hydra)
  - Gin object and gin file managed by [gin-config](https://github.com/google/gin-config)
  - Python module and python file managed by [py_config_runner](https://github.com/vfdev-5/py_config_runner)

- Automatic mixed precision training
  - In this example we use [nvidia/apex](https://github.com/NVIDIA/apex/) (if installed)
  - Native AMP is available in PyTorch since 1.6.0
    - Please, see [our benchmark](https://github.com/pytorch/ignite/blob/master/examples/notebooks/Cifar100_bench_amp.ipynb) of two approaches
  
- [Gradients accumulation](https://pytorch.org/ignite/faq.html#gradients-accumulation) in training step
