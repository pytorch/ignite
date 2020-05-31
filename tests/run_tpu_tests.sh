#!/bin/bash

set -xeu


if [ -z ${NUM_TPU_WORKERS+x} ]; then
    export NUM_TPU_WORKERS=1
    py.test --cov ignite --cov-report xml tests/ -vvv -m tpu
fi

py.test --cov ignite --cov-report xml tests/ -vvv -m tpu