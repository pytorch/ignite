#!/bin/bash

set -xeu

py.test --cov ignite --cov-append --cov-report term-missing tests/ -vvv -m tpu
