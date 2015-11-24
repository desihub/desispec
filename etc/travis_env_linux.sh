#!/bin/bash

# Install conda
wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p $HOME/miniconda
export PATH=/home/travis/miniconda/bin:$PATH
conda update --yes conda

# Installation of non-Python dependencies for documentation is now
# in .travis.yml

# Install Python dependencies
source "$( dirname "${BASH_SOURCE[0]}" )"/travis_env_common.sh
