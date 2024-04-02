#!/bin/bash

set -xeu

pytest --cov ignite --cov-report term-missing --cov-report xml tests/ -vvv -m tpu "${EXTRA_PYTEST_ARGS:-}"

if [ -z ${NUM_TPU_WORKERS+x} ]; then
    export NUM_TPU_WORKERS=1
    pytest --cov ignite --cov-append --cov-report term-missing --cov-report xml tests/ -vvv -m tpu "${EXTRA_PYTEST_ARGS:-}"
fi
