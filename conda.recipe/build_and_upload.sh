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

# Set version in meta.yaml
version=$(sed -nE 's/__version__ = "(.*)"/\1/p' ignite/__init__.py)
sed -i "s/__version__ = \"\(.*\)\"/__version__ = \"$version\"/g" conda.recipe/meta.yaml
cat conda.recipe/meta.yaml | grep version

conda install -y conda-build conda-verify anaconda-client conda-package-handling
conda config --set anaconda_upload no

conda build --no-test --output-folder conda_build conda.recipe -c pytorch --package-format 1
cph transmute $(ls conda_build/*/*.tar.bz2) .conda

# Upload to Anaconda
conda config --set anaconda_upload yes
ls conda_build/*/*.{conda,tar.bz2} | xargs -I {} anaconda -v -t $ANACONDA_TOKEN upload -u $UPLOAD_USER {}
