#!/bin/bash

echo "Build all MS DeepSpeed flavoured PyTorch-Ignite images"

# Start script from ignite docker folder
if [ ! -d msdp ]; then
    echo "Can not find 'msdp' folder"
    echo "Usage: sh msdp/build_all.sh"
    exit 1
fi

curr_dir=$PWD
cd $curr_dir/msdp

set -xeu

image_name="msdp-apex-base"

docker build -t pytorchignite/${image_name}:latest -f Dockerfile.${image_name} .
image_tag=`docker run --rm -i pytorchignite/${image_name}:latest -c 'python -c "import torch; import ignite; print(torch.__version__ + \"-\" + ignite.__version__, end=\"\")"'`
docker tag pytorchignite/${image_name}:latest pytorchignite/${image_name}:${image_tag}

for image_name in "msdp-apex-vision" "msdp-apex-nlp"
do

    docker build -t pytorchignite/${image_name}:latest -f Dockerfile.${image_name} .
    docker tag pytorchignite/${image_name}:latest pytorchignite/${image_name}:${image_tag}

done

cd $curr_dir
