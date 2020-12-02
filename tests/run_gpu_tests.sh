#!/bin/bash

if [ -z "$1" ]; then
    ngpus=1
else
    ngpus=$1
fi

set -xeu

pytest --cov ignite --cov-report term-missing --cov-report xml -vvv tests/ -k 'on_cuda'

# https://pubs.opengroup.org/onlinepubs/009695399/utilities/xcu_chap02.html#tag_02_06_02
if [ "${SKIP_DISTRIB_TESTS:-0}" -eq "1" ]; then
    exit 0
fi

pytest --cov ignite --cov-append --cov-report term-missing --cov-report xml -vvv tests/ -m distributed


if [ ${ngpus} -gt 1 ]; then

    export WORLD_SIZE=${ngpus}
    pytest --cov ignite --cov-append --cov-report term-missing --cov-report xml --dist=each --tx ${WORLD_SIZE}*popen//python=python3.7 tests -m distributed -vvv

fi
