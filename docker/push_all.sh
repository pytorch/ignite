#!/bin/bash

echo "Push all PyTorch-Ignite docker images"

if [ -z $DOCKER_USER ]; then
    echo "Can not find DOCKER_USER env variable"
    echo "Please, export DOCKER_USER=<username> before calling this script"
    exit 1
fi

if [ -z $DOCKER_TOKEN ]; then
    echo "Can not find DOCKER_TOKEN env variable"
    echo "Please, export DOCKER_TOKEN=<token> before calling this script"
    exit 1
fi

set -eu

echo $DOCKER_TOKEN | docker login --username=$DOCKER_USER --password-stdin

set -xeu

image_name="base"
image_tag=`docker run --rm -it pytorchignite/${image_name}:latest -c 'python -c "import torch; import ignite; print(torch.__version__ + \"-\" + ignite.__version__, end=\"\")"'`

for image_name in "base" "vision" "nlp" "apex" "apex-vision" "apex-nlp"
do

    docker push pytorchignite/${image_name}:latest
    docker push pytorchignite/${image_name}:${image_tag}

done


image_name="hvd-base"
image_tag=`docker run --rm -it pytorchignite/${image_name}:latest -c 'python -c "import torch; import ignite; print(torch.__version__ + \"-\" + ignite.__version__, end=\"\")"'`

for image_name in "hvd-base" "hvd-vision" "hvd-nlp" "hvd-apex" "hvd-apex-vision" "hvd-apex-nlp"
do

    docker push pytorchignite/${image_name}:latest
    docker push pytorchignite/${image_name}:${image_tag}

done

image_name="msdp-apex-base"
image_tag=`docker run --rm -it pytorchignite/${image_name}:latest -c 'python -c "import torch; import ignite; print(torch.__version__ + \"-\" + ignite.__version__, end=\"\")"'`

for image_name in "msdp-apex-base" "msdp-apex-vision" "msdp-apex-nlp"
do

    docker push pytorchignite/${image_name}:latest
    docker push pytorchignite/${image_name}:${image_tag}

done

# If use locally, mind to clean dangling images
# docker images | grep 'pytorchignite\|<none>' | awk '{print $3}' | xargs docker rmi -f
# or
# docker image prune
