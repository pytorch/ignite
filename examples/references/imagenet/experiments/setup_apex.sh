#!/bin/bash


tmp_apex_path="/tmp/apex"

python -c "import apex" 2>&1 > /dev/null
res=$?

if [ "$res" -eq "1" ]; then

    echo "Setup NVIDIA Apex"
    git clone https://github.com/NVIDIA/apex $tmp_apex_path
    cd $tmp_apex_path
    pip install --upgrade --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

fi
