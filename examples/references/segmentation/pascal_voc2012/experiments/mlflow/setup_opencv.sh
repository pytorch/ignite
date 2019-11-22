#!/bin/bash


python -c "import cv2"
res=$?

if [ "$res" -eq "1" ]; then
    echo "Install libglib2.0 for opencv"
    apt-get update
    apt-get -y install --no-install-recommends libglib2.0
fi
