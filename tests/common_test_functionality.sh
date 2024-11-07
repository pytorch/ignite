#!/bin/bash

# Will catch exit code 5 when tests are deselected from previous passing run
# (relevent for --last-failed-no-failures none)
last_failed_no_failures_code=5

#  functions shared across test files
run_tests() {
    # Set defaults
    local core_args="-vvv tests/ignite"
    local cache_dir=".unknown-cache"
    local skip_distrib_tests=1
    local match_tests_expression=""
    local trap_deselected_exit_code=1
    local use_last_failed=0
    local use_coverage=0
    local world_size=0
    # Always clean up pytest.ini
    trap 'rm -f pytest.ini' RETURN
    # Parse arguments
    while [[ $# -gt 0 ]]
    do
        key="$1"
        case $key in
            --core_args)
            core_args="$2"
            shift
            shift
            ;;
            --cache_dir)
            cache_dir="$2"
            shift
            shift
            ;;
            --skip_distrib_tests)
            skip_distrib_tests="$2"
            shift
            shift
            ;;
            --match_tests_expression)
            match_tests_expression="$2"
            shift
            shift
            ;;
            --trap_deselected_exit_code)
            trap_deselected_exit_code="$2"
            shift
            shift
            ;;
            --use_last_failed)
            use_last_failed="$2"
            shift
            shift
            ;;
            --use_coverage)
            use_coverage="$2"
            shift
            shift
            ;;
            --world_size)
            world_size="$2"
            shift
            shift
            ;;
            *)
            echo "Error: Unknown argument $key"
            exit 1
            shift
            ;;
        esac
    done

    if ! command -v pytest &> /dev/null
    then
        echo "pytest could not be found"
        echo "The path is: ${PATH}"
        exit 1
    fi


    if [ "${skip_distrib_tests}" -eq "1" ]; then
        # can be overwritten by core_args
        skip_distrib_opt="-m 'not distributed and not tpu and not multinode_distributed'"
    else
        skip_distrib_opt=""
    fi

    echo [pytest] > pytest.ini ; echo "cache_dir=${cache_dir}" >> pytest.ini

    # Assemble options for the pytest command
    pytest_args="${skip_distrib_opt} ${core_args} --treat-unrun-as-failed -k '${match_tests_expression}'"
    if [ "${use_last_failed:-0}" -eq "1" ] && [ -d "${cache_dir}" ]; then
        pytest_args="--last-failed --last-failed-no-failures none ${pytest_args}"
    fi
    if [ "${use_coverage}" -eq "1" ]; then
        pytest_args="--cov ignite --cov-append --cov-report term-missing --cov-report xml ${pytest_args}"
    fi
    if [ ! "${world_size}" -eq "0" ]; then
        export WORLD_SIZE="${world_size}"
        pytest_args="--dist=each --tx ${WORLD_SIZE}*popen//python=python ${pytest_args}"
    fi

    # Run the command
    if [ "$trap_deselected_exit_code" -eq "1" ]; then
        eval "pytest ${pytest_args}" || { exit_code=$?; if [ "$exit_code" -eq ${last_failed_no_failures_code} ]; then echo "All tests deselected"; else exit $exit_code; fi; }
    else
        eval "pytest ${pytest_args}"
    fi
}
