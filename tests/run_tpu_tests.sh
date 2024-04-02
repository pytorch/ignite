#!/bin/bash
# Will catch exit code 5 when tests are deselected from previous passing run
EXIT_CODE_ALL_TESTS_DESELECTED=5

set -xeu

CACHE_DIR=.tpu
echo [pytest] > pytest.ini ; echo "cache_dir=${CACHE_DIR}" >> pytest.ini
PYTEST_ARGS="--cov ignite --cov-report term-missing --cov-report xml tests/ -vvv -m tpu"
if [ "${USE_LAST_FAILED:-0}" -eq "1" ] && [ -d "${CACHE_DIR}" ]; then
    PYTEST_ARGS="--last-failed --last-failed-no-failures none ${PYTEST_ARGS}"
fi
CUDA_VISIBLE_DEVICES="" eval "pytest ${PYTEST_ARGS}" || { exit_code=$?; if [ "$exit_code" -eq 5 ]; then echo "All tests deselected"; else exit $exit_code; fi;}


if [ -z ${NUM_TPU_WORKERS+x} ]; then
    export NUM_TPU_WORKERS=1
    CACHE_DIR=.tpu-multi
    echo [pytest] > pytest.ini ; echo "cache_dir=${CACHE_DIR}" >> pytest.ini
    PYTEST_ARGS="--cov ignite --cov-append --cov-report term-missing --cov-report xml tests/ -vvv -m tpu"
    if [ "${USE_LAST_FAILED:-0}" -eq "1" ] && [ -d "${CACHE_DIR}" ]; then
        PYTEST_ARGS="--last-failed --last-failed-no-failures none ${PYTEST_ARGS}"
    fi
    CUDA_VISIBLE_DEVICES="" eval "pytest ${PYTEST_ARGS}" || { exit_code=$?; if [ "$exit_code" -eq 5 ]; then echo "All tests deselected"; else exit $exit_code; fi;}
fi
rm -f pytest.ini
