# Template examples with Ignite

This folder contains template code examples ranged by domain to quick start with Ignite.

* [x] Vision (Image Processing task)

* [ ] NLP (coming soon)


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
