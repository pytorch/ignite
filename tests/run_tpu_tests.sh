#!/bin/bash

set -xeu

# Catch exit code 5 when tests are deselected from previous passing run
pytest ${EXTRA_PYTEST_ARGS:-} --cov ignite --cov-report term-missing --cov-report xml tests/ -vvv -m tpu || { exit_code=$?; if [ "$exit_code" -eq 5 ]; then echo "All tests deselected"; else exit $exit_code; fi;}

if [ -z ${NUM_TPU_WORKERS+x} ]; then
    export NUM_TPU_WORKERS=1
    pytest ${EXTRA_PYTEST_ARGS:-} --cov ignite --cov-append --cov-report term-missing --cov-report xml tests/ -vvv -m tpu
fi
