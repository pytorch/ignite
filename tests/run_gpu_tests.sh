#!/bin/bash

if [ -z "$1" ]; then
    ws=1
else
    ws=$1
fi

set -xeu

py.test -vvv tests/ -k 'on_cuda'

if [ "$ws" -eq "1" ]; then

    py.test -vvv tests/ -m distributed

else

    export WORLD_SIZE=$ws
    py.test --dist=each --tx $WORLD_SIZE*popen//python=python3.7 tests -m distributed -vvv

fi