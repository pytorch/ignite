#!/bin/bash

if [ -z "$1" ]; then
    ws=1
else
    ws=$1
fi

set -xeu

py.test --cov ignite --cov-report term-missing -vvv tests/ -k 'on_cuda'

export WORLD_SIZE=$ws

py.test --cov ignite --cov-append --cov-report term-missing --dist=each --tx $WORLD_SIZE*popen//python=python3.7 tests -m distributed -vvv
