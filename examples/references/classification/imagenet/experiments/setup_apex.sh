#!/bin/bash


tmp_apex_path="/tmp/apex"

python -c "import apex"
res=$?

if [ "$res" -eq "1" ]; then

    echo "Setup NVIDIA Apex"
    rm -rf $tmp_apex_path
    git clone https://github.com/NVIDIA/apex $tmp_apex_path
    cd $tmp_apex_path
    export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"
    pip install --upgrade --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

fi
