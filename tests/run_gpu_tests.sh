#!/bin/bash

if [ -z "$1" ]; then
    ngpus=1
else
    ngpus=$1
fi

set -xeu

py.test --cov ignite --cov-report term-missing --cov-report xml -vvv tests/ -k 'on_cuda'

if [ "${ngpus}" -eq "1" ]; then

    py.test --cov ignite --cov-append --cov-report term-missing --cov-report xml -vvv tests/ -m distributed

else

    py.test --cov ignite --cov-append --cov-report term-missing --cov-report xml -vvv tests/ -m distributed

    export WORLD_SIZE=${ngpus}
    py.test --cov ignite --cov-append --cov-report term-missing --cov-report xml --dist=each --tx ${WORLD_SIZE}*popen//python=python3.7 tests -m distributed -vvv

fi