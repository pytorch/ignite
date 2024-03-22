#!/bin/bash

set -xeu

if [ $1 = "lint" ]; then
    flake8 ignite tests examples --config setup.cfg
    ufmt diff .
elif [ $1 = "fmt" ]; then
    ufmt format .
elif [ $1 = "mypy" ]; then
    mypy --config-file mypy.ini
elif [ $1 = "install" ]; then
    pip install --upgrade flake8 "black==24.3.0" "usort==1.0.8.post1" "ufmt==2.5.1" "mypy"
fi
