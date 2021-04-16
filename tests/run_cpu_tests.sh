#!/bin/bash

set -xeu

if [ "${SKIP_DISTRIB_TESTS:-0}" -eq "1" ]; then
    skip_distrib_opt=(-m "not distributed and not tpu and not multinode_distributed")
else
    skip_distrib_opt=(-m "")
fi
# To avoid flaky tests retry if fail for a max of 3 times with 2 seconds in between
n=0
until [ "$n" -ge 3 ]
do
   CUDA_VISIBLE_DEVICES="" pytest --tx 4*popen//python=python --cov ignite --cov-report term-missing --cov-report xml -vvv tests "${skip_distrib_opt[@]}" && break
   n=$((n+1))
   sleep 2
done

# https://pubs.opengroup.org/onlinepubs/009695399/utilities/xcu_chap02.html#tag_02_06_02
if [ "${SKIP_DISTRIB_TESTS:-0}" -eq "1" ]; then
    exit 0
fi

export WORLD_SIZE=2
# To avoid flaky tests retry if fail for a max of 3 times with 2 seconds in between
n=0
until [ "$n" -ge 3 ]
do
   CUDA_VISIBLE_DEVICES="" pytest --cov ignite --cov-append --cov-report term-missing --cov-report xml --dist=each --tx $WORLD_SIZE*popen//python=python tests -m distributed -vvv && break
   n=$((n+1))
   sleep 2
done
unset WORLD_SIZE
