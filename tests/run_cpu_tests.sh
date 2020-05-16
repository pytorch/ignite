#!/bin/bash

set -xeu

CUDA_VISIBLE_DEVICES="" py.test --tx 4*popen//python=python$CI_PYTHON_VERSION --cov ignite --cov-report term-missing -vvv tests/

export WORLD_SIZE=2

CUDA_VISIBLE_DEVICES="" py.test --cov ignite --cov-append --cov-report term-missing --dist=each --tx $WORLD_SIZE*popen//python=python$CI_PYTHON_VERSION tests -m distributed -vvv 
