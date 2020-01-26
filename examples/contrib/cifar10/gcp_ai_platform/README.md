# Distributed training on CIFAR10 with GCP AI Platform

In this folder, there are helper scripts to run distributed training on [GCP AI Platform](https://cloud.google.com/ml-engine/docs/).

To use the scripts user needs to have
- an account on GCP and enabled AI Platform, see [here for details](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-keras#set_up_your_project)
- `gcloud` installed and properly configured
- `docker`

## Setup 

- Create output bucket
```bash
gsutil mb -p your-project -l region gs://output-cifar10/ 
# e.g. gsutil mb -p ignite-distrib -l us-east1 gs://output-cifar10/ 
```

- Configure local docker to push to GCR

```bash
gcloud auth configure-docker
```

## Start training

By default, we use `n1-standard-4` and `nvidia-tesla-k80` for the training. For other configs, please edit `submit_job.sh`.
To start training, simply execute :
```bash
sh gcp_ai_platform/submit_job.sh your-project region num_nodes num_gpus_per_node
# sh gcp_ai_platform/submit_job.sh ignite-distrib us-east1 1 2
```

### Training options modifications

Training command line is defined in `entrypoint.sh`. To add/remove options, user should edit this file.

### Logs visualization - Tensorboard

By default, AI platform provides stream logs in the web interface, see docs [here](https://cloud.google.com/ml-engine/docs/monitor-training#checking_job_status).
In addition, user can setup [Tensorboard locally or from Cloud Shell](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction#tensorboard-local) : 
```
# in cloud shell
tensorboard --logdir=gs://output-cifar10
```

