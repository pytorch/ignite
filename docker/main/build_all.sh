#!/bin/bash

echo "Build all PyTorch-Ignite images"

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

# check OpenCV import is valid
check_opencv()
{
  image_to_check=$1
  { # try
    msg=`docker run --rm -it ${image_to_check} python -c "import cv2"`  &&
    echo "OpenCV import OK"
  } || { # catch
      echo "${msg}"
      echo "OpenCV import NOK : removing ${image_to_check}"
      docker rmi -f ${image_to_check}
      exit 1
  }
}

# Start script from ignite docker folder
if [ ! -d main ]; then
    echo "Can not find 'main' folder"
    echo "Usage: sh main/build_all.sh"
    exit 1
fi

curr_dir=$PWD
cd $curr_dir/main

set -eu

image_tag=""

if [ -z "${PTH_VERSION}" ]; then
    echo "PTH_VERSION is not set"
    exit 1
fi

pth_version=${PTH_VERSION}

for image_name in "base" "vision" "nlp" "apex" "apex-vision" "apex-nlp"
do

    retry "docker build --build-arg PTH_VERSION=${pth_version} -t pytorchignite/${image_name}:latest -f Dockerfile.${image_name} ." "\nBuild failed: ${image_name}"
    if [ -z $image_tag ]; then
        image_tag=`docker run --rm -i pytorchignite/${image_name}:latest python -c "import torch; import ignite; print(torch.__version__ + \"-\" + ignite.__version__, end=\"\")"`
    fi
    if [ ${image_name} = "vision" ] || [ ${image_name} = "apex-vision" ]; then
        check_opencv "pytorchignite/${image_name}:latest"
    fi

    docker tag pytorchignite/${image_name}:latest pytorchignite/${image_name}:${image_tag}

done

cd $curr_dir
