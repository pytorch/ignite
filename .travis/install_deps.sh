#!/bin/sh
sudo apt-get update
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
  # Useful for debugging any issues with conda
conda info -a
conda create -q -n test-environment pytorch cpuonly python=$TRAVIS_PYTHON_VERSION -c $PYTORCH_CHANNEL
source activate test-environment
# Keep fix in case of problem with torchvision nightly releases
# - if [[ "$PYTORCH_CHANNEL" == "pytorch-nightly" ]]; then pip install --upgrade git+https://github.com/pytorch/vision.git; else conda install torchvision cpuonly python=$TRAVIS_PYTHON_VERSION -c $PYTORCH_CHANNEL; fi
conda install torchvision cpuonly python=$TRAVIS_PYTHON_VERSION -c $PYTORCH_CHANNEL
# Install all test/examples dependencies
pip install -r requirements-dev.txt
