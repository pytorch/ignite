#!/bin/bash

set -xeu

py.test --cov ignite --cov-report xml tests/ -vvv -m tpu
