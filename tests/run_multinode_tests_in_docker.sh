#!/bin/bash

# Tests configuration:
export nnodes=2
export nproc_per_node=4
export gpu=0

# Start script from ignite root folder
if [ ! -d tests ]; then
    echo "Ignite tests folder is not found. Please run script from ignite's root folder"
    exit 1
fi

docker_image="pytorch/pytorch:latest"
install_test_requirements="pip install mock pytest pytest-xdist scikit-learn"
cmd="pytest --dist=each --tx $nproc_per_node*popen//python=python3.6 tests -m multinode_distributed -vvv $@"


export MASTER_ADDR=node0
export MASTER_PORT=9999

network=tempnet

# Create user bridge network
docker network create --driver bridge $network


if [ $gpu == 1 ]; then
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

    docker run --rm $is_detached \
               -v $PWD:/workspace $env_multinode_option \
               --env nnodes \
               --env nproc_per_node \
               --env node_id \
               --env MASTER_ADDR \
               --env MASTER_PORT \
               --name $node_name \
               --network $network \
               $docker_image /bin/bash -c "$install_test_requirements && $cmd"

done

sleep 5

docker network rm $network
