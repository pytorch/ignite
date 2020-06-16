#!/bin/bash


python -c "import cv2"
res=$?

if [ "$res" -eq "1" ]; then
    echo "Install libglib2.0 libsm6 libxext6 libxrender-dev for opencv"
    apt-get update
    ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
    apt-get install -y tzdata
    dpkg-reconfigure --frontend noninteractive tzdata
    apt-get -y install --no-install-recommends libglib2.0 libsm6 libxext6 libxrender-dev
fi
