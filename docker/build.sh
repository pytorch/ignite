#!/bin/bash

if [ -z "$1" ]; then
  echo "Folder name should be provided. Usage, for example: bash build.sh hvd hvd-apex"
  exit 1
fi

if [ -z "$2" ]; then
  echo "Image name should be provided. Usage, for example: bash build.sh hvd hvd-apex"
  exit 1
fi


folder_name=$1
image_name=$2

echo "Build ${folder_name}/Dockerfile.${image_name} PyTorch-Ignite image"

# Start script from ignite docker folder
if [ ! -d ${folder_name} ]; then
    echo "Can not find ${folder_name} folder"
    exit 1
fi

# Check Dockerfile exists
if [ ! -f ${folder_name}/Dockerfile.${image_name} ]; then
    echo "Can not find ${folder_name}/Dockerfile.${image_name}"
    exit 1
fi

if [ -z "${PTH_VERSION}" ]; then
    echo "PTH_VERSION is not set. Please, call the script with: PTH_VERSION=... bash build.sh hvd hvd-apex"
    exit 1
fi

if [ ${folder_name} == "hvd" ] && [ -z "${HVD_VERSION}" ]; then
    echo "HVD_VERSION is not set"
    exit 1
fi

if [ ${folder_name} == "msdp" ] && [ -z "${MSDP_VERSION}" ]; then
    echo "MSDP_VERSION is not set"
    exit 1
fi


retry()
{
    cmd=$1 msg=$2
    counter=0 limit=3
    while [ "$counter" -lt "$limit" ]; do
        echo "(Re-)Try: $cmd"
        bash -c "$cmd" && break
        echo $msg
        counter="$(( $counter + 1 ))"
    done
    if [ $counter -eq $limit ]; then
        exit 1
    fi
}

curr_dir=$PWD
cd $curr_dir/${folder_name}

set -eu

image_tag=""
pth_version=${PTH_VERSION}
opt_build_args=""

if [ -n "${HVD_VERSION:-}" ]; then
    opt_build_args="${opt_build_args} --build-arg HVD_VERSION=${HVD_VERSION}"
fi

if [ -n "${MSDP_VERSION:-}" ]; then
    opt_build_args="${opt_build_args} --build-arg MSDP_VERSION=${MSDP_VERSION}"
fi

echo "opt_build_args: ${opt_build_args}"

retry "docker build --build-arg PTH_VERSION=${pth_version} ${opt_build_args} -t pytorchignite/${image_name}:latest -f Dockerfile.${image_name} ." "\nBuild failed: ${image_name}"
if [ -z $image_tag ]; then
    image_tag=`docker run --rm -i pytorchignite/${image_name}:latest python -c "import torch; import ignite; print(torch.__version__.split('+')[0] + \"-\" + ignite.__version__, end=\"\")"`
fi
docker tag pytorchignite/${image_name}:latest pytorchignite/${image_name}:${image_tag}

cd $curr_dir

# Test built image
echo "Show installed packages:"
docker run --rm -i pytorchignite/${image_name}:${image_tag} pip list

echo "Test pytorchignite/${image_name}:${image_tag}"
docker run --rm -i -v $PWD:/ws -w /ws -e HVD_VERSION=${HVD_VERSION:-} -e MSDP_VERSION=${MSDP_VERSION:-} pytorchignite/${image_name}:${image_tag} /bin/bash -c "python test_image.py pytorchignite/${image_name}:${image_tag}"
echo "OK"
