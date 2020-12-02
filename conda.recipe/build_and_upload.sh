#!/bin/bash

echo "Build and upload Conda binaries"

# ANACONDA_TOKEN should be provided
# How to generate ANACONDA_TOKEN: https://docs.anaconda.com/anaconda-cloud/user-guide/tasks/work-with-accounts#creating-access-tokens
# https://conda.io/docs/user-guide/tasks/build-packages/install-conda-build.html

if [ -z $ANACONDA_TOKEN ]; then
    echo "Can not find ANACONDA_TOKEN env variable"
    echo "Please, export ANACONDA_TOKEN=<username> before calling this script"
    exit 1
fi

if [ -z $UPLOAD_USER ]; then
    echo "Can not find UPLOAD_USER env variable"
    echo "Please, export UPLOAD_USER=<username> before calling this script"
    exit 1
fi

set -xeu

conda install -y conda-build conda-verify anaconda-client
conda config --set anaconda_upload no

conda build --no-test --output-folder conda_build conda.recipe -c pytorch

# Upload to Anaconda
# We could use --all but too much platforms to uploaded
ls conda_build/*/*.tar.bz2 | xargs -I {} anaconda -v -t $ANACONDA_TOKEN upload -u $UPLOAD_USER {}
