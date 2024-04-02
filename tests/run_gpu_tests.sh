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

# Will catch exit code 5 when tests are deselected from previous passing run
EXIT_CODE_ALL_TESTS_DESELECTED=5

set -xeu

CACHE_DIR=.gpu-cuda
echo [pytest] > pytest.ini ; echo "cache_dir=${CACHE_DIR}" >> pytest.ini
PYTEST_ARGS="--cov ignite --cov-report term-missing --cov-report xml -vvv tests/ -k '${cuda_pattern}'"
if [ "${USE_LAST_FAILED:-0}" -eq "1" ] && [ -d "${CACHE_DIR}" ]; then
    PYTEST_ARGS="--last-failed --last-failed-no-failures none ${PYTEST_ARGS}"
fi
CUDA_VISIBLE_DEVICES="" eval "pytest ${PYTEST_ARGS}" || { exit_code=$?; if [ "$exit_code" -eq 5 ]; then echo "All tests deselected"; else exit $exit_code; fi;}



# https://pubs.opengroup.org/onlinepubs/009695399/utilities/xcu_chap02.html#tag_02_06_02
if [ "${SKIP_DISTRIB_TESTS:-0}" -eq "1" ]; then
    exit 0
fi

CACHE_DIR=.gpu-distrib
echo [pytest] > pytest.ini ; echo "cache_dir=${CACHE_DIR}" >> pytest.ini
PYTEST_ARGS="--cov ignite --cov-append --cov-report term-missing --cov-report xml -vvv tests/ -m distributed -k '${MATCH_TESTS_EXPRESSION}'"
if [ "${USE_LAST_FAILED:-0}" -eq "1" ] && [ -d "${CACHE_DIR}" ]; then
    PYTEST_ARGS="--last-failed --last-failed-no-failures none ${PYTEST_ARGS}"
fi
CUDA_VISIBLE_DEVICES="" eval "pytest ${PYTEST_ARGS}" || { exit_code=$?; if [ "$exit_code" -eq 5 ]; then echo "All tests deselected"; else exit $exit_code; fi;}

if [ ${ngpus} -gt 1 ]; then

    export WORLD_SIZE=${ngpus}
    CACHE_DIR=.gpu-distrib-multi
    echo [pytest] > pytest.ini ; echo "cache_dir=${CACHE_DIR}" >> pytest.ini
    PYTEST_ARGS="--cov ignite --cov-append --cov-report term-missing --cov-report xml --dist=each --tx ${WORLD_SIZE}*popen//python=python tests -m distributed -vvv -k '${MATCH_TESTS_EXPRESSION}'"
    if [ "${USE_LAST_FAILED:-0}" -eq "1" ] && [ -d "${CACHE_DIR}" ]; then
        PYTEST_ARGS="--last-failed --last-failed-no-failures none ${PYTEST_ARGS}"
    fi
    CUDA_VISIBLE_DEVICES="" eval "pytest ${PYTEST_ARGS}"
    unset WORLD_SIZE

fi
rm -f pytest.ini
