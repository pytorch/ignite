#!/bin/bash

# Tests configuration:
if [[ -z "$1" || "$1" -lt 2 ]]; then
    echo "nnodes setting default to 2"
    export nnodes=2
else
    export nnodes=$1
fi

if [[ -z "$2" || "$2" -lt 1 ]]; then
    echo "nproc_per_node setting default to 4"
    export nproc_per_node=4
else
    export nproc_per_node=$2
fi

if [ -z "$3" ]; then
    echo "gpu setting default to 0 ( False )"
    export gpu=0
else
    export gpu=$3
fi

# Start script from ignite root folder
if [ ! -d tests ]; then
    echo "Ignite tests folder is not found. Please run script from ignite's root folder"
    exit 1
fi

docker_image="pytorchignite/tests:latest"

docker build -t $docker_image -<<EOF
FROM pytorch/pytorch:latest
RUN pip install --no-cache-dir mock pytest pytest-xdist scikit-learn scikit-image dill matplotlib clearml
EOF

docker_python_version=`docker run --rm -i $docker_image python -c "import sys; print(str(sys.version_info[0]) + \".\" + str(sys.version_info[1]), end=\"\")"`
cmd="pytest --dist=each --tx $nproc_per_node*popen//python${docker_python_version} -m multinode_distributed -vvv tests/ignite"

export MASTER_ADDR=node0
export MASTER_PORT=9999

network=tempnet

# Create user bridge network
docker network create --driver bridge $network


if [ $gpu -gt 0 ]; then
    env_multinode_option="-e GPU_MULTINODE_DISTRIB=1"
else
    env_multinode_option="-e MULTINODE_DISTRIB=1"
fi


for i in $(seq 0 $((nnodes - 1)) )
do

    echo "Start Node $i"
    node_name="node$i"

    is_detached="-d"
    if [ $i == $((nnodes - 1)) ]; then
        is_detached=""
    fi

    export node_id=$i

    if [ $gpu -gt 0 ]; then
      gpu_options="--gpus device=$i"
    else
      gpu_options=""
    fi

    docker run $is_detached $gpu_options \
               -v $PWD:/workspace $env_multinode_option \
               --env nnodes \
               --env nproc_per_node \
               --env node_id \
               --env MASTER_ADDR \
               --env MASTER_PORT \
               --name $node_name \
               --network $network \
               $docker_image /bin/bash -c "$cmd"

done

sleep 5

for i in $(seq 0 $((nnodes - 1)) )
do
    echo "Removing Node $i"
    node_name="node$i"
    docker rm $node_name
done

docker network rm $network
