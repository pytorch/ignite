#!/bin/bash

if [ -z "$1" ]; then
    ngpus=1
else
    ngpus=$1
fi

MATCH_TESTS_EXPRESSION=${2:-""}

if [ -z "$MATCH_TESTS_EXPRESSION" ]; then
    cuda_pattern="cuda"
else
    cuda_pattern="cuda and $MATCH_TESTS_EXPRESSION"
fi

set -xeu

pytest --cov ignite --cov-report term-missing --cov-report xml -vvv tests/ "${EXTRA_PYTEST_ARGS:-}" -k "$cuda_pattern"

# https://pubs.opengroup.org/onlinepubs/009695399/utilities/xcu_chap02.html#tag_02_06_02
if [ "${SKIP_DISTRIB_TESTS:-0}" -eq "1" ]; then
    exit 0
fi

pytest --cov ignite --cov-append --cov-report term-missing --cov-report xml -vvv tests/ -m distributed "${EXTRA_PYTEST_ARGS:-}" -k "$MATCH_TESTS_EXPRESSION"


if [ ${ngpus} -gt 1 ]; then

    export WORLD_SIZE=${ngpus}
    pytest --cov ignite --cov-append --cov-report term-missing --cov-report xml --dist=each --tx ${WORLD_SIZE}*popen//python=python tests -m distributed -vvv "${EXTRA_PYTEST_ARGS:-}" -k "$MATCH_TESTS_EXPRESSION"
    unset WORLD_SIZE

fi
