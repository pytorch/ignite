#!/bin/bash
source "$(dirname "$0")/common_test_functionality.sh"
set -xeu

skip_distrib_tests=${SKIP_DISTRIB_TESTS:-0}
use_last_failed=${USE_LAST_FAILED:-0}
match_tests_expression=${1:-""}

use_xdist=${USE_XDIST:-1}
core_args="-vvv tests/ignite"
if [ "${use_xdist}" -eq "1" ]; then
    core_args="${core_args} --tx 4*popen//python=python"
fi

CUDA_VISIBLE_DEVICES="" run_tests \
    --core_args "${core_args}" \
    --cache_dir ".cpu-not-distrib" \
    --skip_distrib_tests "${skip_distrib_tests}" \
    --use_coverage 1 \
    --match_tests_expression "${match_tests_expression}" \
    --use_last_failed ${use_last_failed}

# https://pubs.opengroup.org/onlinepubs/009695399/utilities/xcu_chap02.html#tag_02_06_02
if [ "${skip_distrib_tests}" -eq "1" ] || [ "${use_xdist}" -eq "0" ]; then
    exit 0
fi

# Run 2 processes with --dist=each
CUDA_VISIBLE_DEVICES="" run_tests \
    --core_args "-m distributed -vvv tests/ignite" \
    --world_size 4 \
    --cache_dir ".cpu-distrib" \
    --skip_distrib_tests 0 \
    --use_coverage 1 \
    --match_tests_expression "${match_tests_expression}" \
    --use_last_failed ${use_last_failed}
