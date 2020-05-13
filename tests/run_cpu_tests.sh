#!/bin/bash

set -xeu

CUDA_VISIBLE_DEVICES="" py.test --tx 4*popen//python=python$TRAVIS_PYTHON_VERSION --cov ignite --cov-report term-missing -vvv tests/

export WORLD_SIZE=2

# Rerun failing tests up to five times with a 30s delay
CUDA_VISIBLE_DEVICES="" py.test --cov ignite --cov-append --cov-report term-missing --dist=each --tx $WORLD_SIZE*popen//python=python$TRAVIS_PYTHON_VERSION tests -m distributed -vvv --reruns 5 --reruns-delay 30
