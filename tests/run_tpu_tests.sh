#!/bin/bash
source "$(dirname "$0")/common_test_functionality.sh"
set -xeu
use_last_failed=${USE_LAST_FAILED:-0}

run_tests \
    --core_args "-vvv -m tpu tests/ignite" \
    --cache_dir ".tpu" \
    --use_coverage 1 \
    --use_last_failed ${use_last_failed}


if [ -z ${NUM_TPU_WORKERS+x} ]; then
    export NUM_TPU_WORKERS=1
    run_tests \
        --core_args "-vvv -m tpu tests/ignite" \
        --cache_dir ".tpu-multi" \
        --use_coverage 1 \
        --use_last_failed ${use_last_failed}
fi
