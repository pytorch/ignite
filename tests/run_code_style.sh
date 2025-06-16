#!/bin/bash

set -xeu

if [ $1 = "lint" ]; then
    uv run flake8 ignite tests examples --config setup.cfg
    uv run ufmt diff .
elif [ $1 = "fmt" ]; then
    uv run ufmt format .
elif [ $1 = "mypy" ]; then
    uv run mypy --config-file mypy.ini
elif [ $1 = "install" ]; then
    uv pip install --upgrade flake8 "black==24.10.0" "usort==1.0.8.post1" "ufmt==2.7.3" "mypy"
fi
