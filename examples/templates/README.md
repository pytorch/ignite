# Template examples with Ignite

This folder contains template code examples ranged by training type to quick start with Ignite.

* [x] [Supervised Single-Model Training](supervised_single_model): vision/nlp tasks like classification

* [ ] [Supervised Multi-Model Training](supervised_multi_model): GANs, Teacher-Student training, etc


## How to use

- Choose your task
- Copy files to your project
- Adapt the code to your specificity
  - dataflow
  - model(s)/optimizer(s)/criterion(s)
  - training step
  - metrics
  - etc

All templates are optionally structured into several files:

- `main.py` - main entry-point to run training script
- `trainer.py` - helper module to setup ignite trainer
- `utils.py` - helper module to setup dataflow, model(s), optimizer(s), criterion(s), lr scheduler(s) etc. 
