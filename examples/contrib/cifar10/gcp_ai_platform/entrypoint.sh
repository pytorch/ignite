#!/bin/bash

echo "\nCLUSTER_SPEC=${CLUSTER_SPEC}\n"

export NUM_NODES=$1
export NUM_GPUS_PER_NODE=$2
export OUTPUT_PATH=$3

cd /workspace/code

# Parse CLUSTER_SPEC
data=`python parse_cluster_spec.py`
data=(${data//,/ })
master_addr=${data[0]}
master_port=${data[1]}
node_rank=${data[2]}

echo "- NUM_NODES=$NUM_NODES"
echo "- NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE"
echo "- OUTPUT_PATH=$OUTPUT_PATH"
echo "- master_addr=$master_addr"
echo "- master_port=$master_port"
echo "- node_rank=$node_rank"


python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --master_addr=$master_addr \
    --master_port=$master_port \
    --node_rank=$node_rank \
    main.py --params="batch_size=512;dist_backend='nccl';data_path=/workspace/data/;output_path=/tmp/output;display_iters=False"


if [ $node_rank == 0 ]; then
    gsutil -m cp -R /tmp/output/ ${OUTPUT_PATH}
fi