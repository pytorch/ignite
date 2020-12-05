#!/bin/bash

set -xeu

if [ "${SKIP_DISTRIB_TESTS:-0}" -eq "1" ]; then
    skip_distrib_opt=(-m "not distributed and not tpu and not multinode_distributed")
else
    skip_distrib_opt=(-m "")
fi

CUDA_VISIBLE_DEVICES="" pytest --tx 4*popen//python=python${CI_PYTHON_VERSION:-3.7} --cov ignite --cov-report term-missing --cov-report xml -vvv tests "${skip_distrib_opt[@]}"

# https://pubs.opengroup.org/onlinepubs/009695399/utilities/xcu_chap02.html#tag_02_06_02
if [ "${SKIP_DISTRIB_TESTS:-0}" -eq "1" ]; then
    exit 0
fi

export WORLD_SIZE=2


CUDA_VISIBLE_DEVICES="" pytest --cov ignite --cov-append --cov-report term-missing --cov-report xml --dist=each --tx $WORLD_SIZE*popen//python=python${CI_PYTHON_VERSION:-3.7} tests -m distributed -vvv
