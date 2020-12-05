#!/bin/bash

set -xeu

pytest --cov ignite --cov-report term-missing --cov-report xml tests/ -vvv -m tpu

if [ -z ${NUM_TPU_WORKERS+x} ]; then
    export NUM_TPU_WORKERS=1
    pytest --cov ignite --cov-report xml tests/ -vvv -m tpu
fi
