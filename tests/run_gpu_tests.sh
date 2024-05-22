#!/bin/bash
source "$(dirname "$0")/common-test-functionality.sh"
set -xeu

skip_distrib_tests=${SKIP_DISTRIB_TESTS:-1}
use_last_failed=${USE_LAST_FAILED:-0}
ngpus=${1:-1}

match_tests_expression=${2:-""}
if [ -z "$match_tests_expression" ]; then
    cuda_pattern="cuda"
else
    cuda_pattern="cuda and $match_tests_expression"
fi

run_tests \
    --core_args "-vvv tests/ignite" \
    --cache_dir ".gpu-cuda" \
    --skip_distrib_tests "${skip_distrib_tests}" \
    --use_coverage 1 \
    --match_tests_expression "${cuda_pattern}" \
    --use_last_failed ${use_last_failed}

# https://pubs.opengroup.org/onlinepubs/009695399/utilities/xcu_chap02.html#tag_02_06_02
if [ "${skip_distrib_tests}" -eq "1" ]; then
    exit 0
fi

run_tests \
    --core_args "-vvv -m distributed tests" \
    --cache_dir ".gpu-distrib" \
    --skip_distrib_tests 0 \
    --use_coverage 1 \
    --match_tests_expression "${match_tests_expression}" \
    --use_last_failed ${use_last_failed}


if [ ${ngpus} -gt 1 ]; then
    run_tests \
        --core_args "-vvv -m distributed tests" \
        --world_size "${ngpus}" \
        --cache_dir ".gpu-distrib-multi" \
        --skip_distrib_tests 0 \
        --use_coverage 1 \
        --match_tests_expression "${match_tests_expression}" \
        --use_last_failed ${use_last_failed}
fi
