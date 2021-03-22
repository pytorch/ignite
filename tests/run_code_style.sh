#!/bin/bash

set -xeu

if [ $1 = "lint" ]; then
    flake8 ignite tests examples --config setup.cfg
    isort . --check --settings setup.cfg
    black . --check --config pyproject.toml
elif [ $1 = "fmt" ]; then
    isort . --settings setup.cfg
    black . --config pyproject.toml
elif [ $1 = "mypy" ]; then
    mypy --config-file mypy.ini
elif [ $1 = "install" ]; then
    pip install flake8 "black==19.10b0" "isort==5.7.0" mypy
fi
