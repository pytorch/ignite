#!/bin/bash

echo "Build all Horovod flavoured PyTorch-Ignite images"

# Start script from ignite docker folder
if [ ! -d hvd ]; then
    echo "Can not find 'hvd' folder"
    echo "Usage: sh hvd/build_all.sh"
    exit 1
fi

curr_dir=$PWD
cd $curr_dir/hvd

set -xeu

image_name="hvd-base"

docker build -t pytorchignite/${image_name}:latest -f Dockerfile.${image_name} .
image_tag=`docker run --rm -i pytorchignite/${image_name}:latest python -c "import torch; import ignite; print(torch.__version__ + \"-\" + ignite.__version__, end=\"\")"`
docker tag pytorchignite/${image_name}:latest pytorchignite/${image_name}:${image_tag}

for image_name in "hvd-vision" "hvd-nlp" "hvd-apex" "hvd-apex-vision" "hvd-apex-nlp"
do

    docker build -t pytorchignite/${image_name}:latest -f Dockerfile.${image_name} .
    docker tag pytorchignite/${image_name}:latest pytorchignite/${image_name}:${image_tag}

done

cd $curr_dir
