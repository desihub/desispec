#!/bin/bash

# Install conda
NOW=`date '+%Y%m%d %T'`
echo "Start install conda at ${NOW}"

wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p $HOME/miniconda
export PATH=$HOME/miniconda/bin:$PATH
conda update --yes conda

NOW=`date '+%Y%m%d %T'`
echo "End install conda at ${NOW}"

# Installation of non-Python dependencies for documentation is now
# in .travis.yml

# Install Python dependencies
NOW=`date '+%Y%m%d %T'`
echo "Start install dependencies at ${NOW}"

source "$( dirname "${BASH_SOURCE[0]}" )"/travis_env_common.sh

NOW=`date '+%Y%m%d %T'`
echo "End install dependencies at ${NOW}"
