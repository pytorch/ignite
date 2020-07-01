# Experiments tracking with Polyaxon

User can run Pascal VOC training using [Polyaxon experiments tracking system](https://polyaxon.com/).

## Requirements

In this case we assume, user has [Polyaxon](https://polyaxon.com/) installed on a machine/cluster/cloud and can schedule experiments with `polyaxon-cli`.

## Usage

### Setup Polyaxon project

Create project on the cluster
```bash
polyaxon project create --name=pascal-voc2012 --description="Semantic segmentation on Pascal VOC2012"
```
Initialize local project
```bash
polyaxon init pascal-voc2012
``` 

Please rename and modify `experiments/plx/xp_training.yml.tmpl` to `experiments/plx/xp_training.yml` 
to adapt to your cluster configuration.

#### Download Pascal VOC 2012 and SBD

Optionally, it is possible to download the datasets as a job. 
Please rename and modify `experiments/plx/job_download_datasets.yml.tmpl` to `experiments/plx/job_download_datasets.yml`
```bash
polyaxon run -u -f experiments/plx/job_download_datasets.yml
```

### Training on single node with single or multiple GPU

For optimal devices usage, please, make sure to adapt training data loader batch size to your infrastructure. 
For example, a single GPU with 11GB can have a batch size of 8-9, thus, on N devices, we can set it as `N * 9`.
Please, adapt `xp_training.yml` to your cluster configuration and run it, for example, as

```bash
polyaxon run -u -f experiments/plx/xp_training.yml --name="baseline_resnet101_sbd" --tags=train,deeplab,sbd
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

File [xp_training.yml.tmpl](experiments/plx/xp_training.yml.tmpl) defines all configurations and dependencies 
necessary for our experimentations. Part `run.cmd` starts single-node multi-GPU training script. 
