# Template example for Image Processing Task (Vision)

## Requirements

```bash
pip install torch torchvision pytorch-ignite
pip install tensorboard
```

## How to use

- Choose your task
- Copy files to your project
- Adapt the code to your specificity
  - dataflow
  - model/optimizer/criterion 
  - training step
  - metrics
  - etc

All templates are optionally structured into several files:

- `main.py` - main entry-point to run training script
- `trainer.py` - helper module to setup ignite trainer
- `utils.py` - helper module to setup dataflow, model, optimizer, criterion, lr scheduler etc. 


### Training on single or multiple GPU(s) 

```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py
```
or
```bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py
```
