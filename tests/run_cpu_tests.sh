#!/bin/bash

set -xeu

CUDA_VISIBLE_DEVICES="" py.test --tx 4*popen//python=python${CI_PYTHON_VERSION:-3.7} --cov ignite --cov-report term-missing -vvv tests/

# https://pubs.opengroup.org/onlinepubs/009695399/utilities/xcu_chap02.html#tag_02_06_02
if [ "${SKIP_DISTRIB_TESTS:-0}" -eq "1" ]; then
    exit 0
fi

export WORLD_SIZE=2

CUDA_VISIBLE_DEVICES="" py.test --cov ignite --cov-append --cov-report term-missing --dist=each --tx $WORLD_SIZE*popen//python=python${CI_PYTHON_VERSION:-3.7} tests -m distributed -vvv
