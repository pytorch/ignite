#!/bin/bash

set -xeu

if [ "${SKIP_DISTRIB_TESTS:-0}" -eq "1" ]; then
    skip_distrib_opt="not distributed and not tpu and not multinode_distributed"
else
    skip_distrib_opt=""
fi

MATCH_TESTS_EXPRESSION=${1:-""}

# Will catch exit code 5 when tests are deselected from previous passing run
EXIT_CODE_ALL_TESTS_DESELECTED=5

CACHE_DIR=.cpu-not-distrib
echo [pytest] > pytest.ini ; echo "cache_dir=${CACHE_DIR}" >> pytest.ini
PYTEST_ARGS="--tx 4*popen//python=python --cov ignite --cov-report term-missing --cov-report xml -vvv tests -m '${skip_distrib_opt}' -k '${MATCH_TESTS_EXPRESSION}'"
if [ "${USE_LAST_FAILED:-0}" -eq "1" ] && [ -d "${CACHE_DIR}" ]; then
    PYTEST_ARGS="--last-failed --last-failed-no-failures none ${PYTEST_ARGS}"
fi
CUDA_VISIBLE_DEVICES="" eval "pytest ${PYTEST_ARGS}"  || { exit_code=$?; if [ "$exit_code" -eq 5 ]; then echo "All tests deselected"; else exit $exit_code; fi;}

# https://pubs.opengroup.org/onlinepubs/009695399/utilities/xcu_chap02.html#tag_02_06_02
if [ "${SKIP_DISTRIB_TESTS:-0}" -eq "1" ]; then
    exit 0
fi

export WORLD_SIZE=2
CACHE_DIR=.cpu-distrib
echo [pytest] > pytest.ini ; echo "cache_dir=${CACHE_DIR}" >> pytest.ini
PYTEST_ARGS="--cov ignite --cov-append --cov-report term-missing --cov-report xml --dist=each --tx ${WORLD_SIZE}*popen//python=python tests -m distributed -vvv -k '${MATCH_TESTS_EXPRESSION}'"
if [ "${USE_LAST_FAILED:-0}" -eq "1" ] && [ -d "${CACHE_DIR}" ]; then
    PYTEST_ARGS="--last-failed --last-failed-no-failures none ${PYTEST_ARGS}"
fi
CUDA_VISIBLE_DEVICES="" eval "pytest ${PYTEST_ARGS}"
unset WORLD_SIZE

rm -f pytest.ini
