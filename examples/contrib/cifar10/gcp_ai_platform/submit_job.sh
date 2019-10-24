#!/bin/bash

if [ -z $4 ]; then
    echo "Usage:"
    echo "\tbash submit_job.sh <PROJECT> <REGION> <NUM_NODES> <NUM_GPUS_PER_NODE>"
    echo "\te.g. bash submit_job.sh ignite-distrib us-east1 2 4"
    exit 0
fi

PROJECT=$1
REGION=$2
NUM_NODES=$3
NUM_GPUS_PER_NODE=$4

JOB_NAME="training-${NUM_NODES}n-${NUM_GPUS_PER_NODE}g-$(date +%Y%m%d-%H%M%S)"
OUTPUT_PATH="gs://output-cifar10/${JOB_NAME}"
IMAGE_URI="gcr.io/$PROJECT/ignite-distrib:latest"

accelerator_type=nvidia-tesla-k80
machine_type=n1-standard-4

echo "\nSetup and start training on $NUM_NODES nodes with $(( NUM_GPUS_PER_NODE * NUM_NODES)) total gpus\n"

set -o errexit
set -o pipefail
set -u
set -x

echo "- Build and push docker images"

docker build -f gcp_ai_platform/Dockerfile -t ${IMAGE_URI} .
docker push ${IMAGE_URI}

job_id=${JOB_NAME//-/_}
echo "- Submit job : $job_id"

workers_setup=""
if [ $(( NUM_NODES - 1 )) -gt 0 ]; then
    workers_setup="--worker-image-uri ${IMAGE_URI} --worker-count $(( NUM_NODES - 1 )) --worker-machine-type $machine_type --worker-accelerator count=$NUM_GPUS_PER_NODE,type=$accelerator_type"
fi


gcloud ai-platform jobs submit training $job_id \
    --project $PROJECT \
    --region $REGION \
    --scale-tier custom \
    --master-image-uri ${IMAGE_URI} \
    --master-machine-type $machine_type \
    --master-accelerator count=$NUM_GPUS_PER_NODE,type=$accelerator_type \
    $workers_setup -- \
    $NUM_NODES $NUM_GPUS_PER_NODE $OUTPUT_PATH
