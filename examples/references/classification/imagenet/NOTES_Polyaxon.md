# Experiments tracking with Polyaxon

## Requirements

In this case we assume, user has [Polyaxon](https://polyaxon.com/) installed on a machine/cluster/cloud and can schedule experiments with `polyaxon-cli`.

## Usage

### Setup Polyaxon project

Create project on the cluster
```
polyaxon project create --name=imagenet --description="Classification on ImageNet"
```
Initialize local project
```
polyaxon init imagenet
``` 

Please rename and modify `experiments/plx/xp_training.yml.tmpl` to `experiments/plx/xp_training.yml` 
to adapt to your cluster configuration.

#### Download ImageNet dataset

Optionally, it is possible to download the datasets as a job. 
Please rename and modify `experiments/plx/job_download_datasets.yml.tmpl` to `experiments/plx/job_download_datasets.yml`
```bash
polyaxon run -u -f experiments/plx/job_download_datasets.yml
```


### Single node with multiple GPUs

```bash
polyaxon run -u -f experiments/plx/xp_training.yml --name="baseline_resnet50" --tags=train,resnet50
```

## Training tracking

Please, see Polyaxon dashboard usage at https://docs.polyaxon.com/


## Implementation details

Files tree description:
```
code
configs  
experiments/plx : Polyaxon related files
notebooks 
```

### Experiments

File [xp_training.yml.tmpl](experiments/mlflow/xp_training.yml.tmpl) defines all configurations and dependencies 
necessary for our experimentations. Part `run.cmd` starts single-node multi-GPU training script. 
