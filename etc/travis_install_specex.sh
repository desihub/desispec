#!/bin/bash -x
#
# SPECEX and HARP.  Once we require specex in any unit tests, we should
# re-enable the installation of harp and specex.
#
export CPATH=$HOME/miniconda/envs/test/include:$CPATH
export LIBRARY_PATH=$HOME/miniconda/envs/test/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/miniconda/envs/test/lib:$LD_LIBRARY_PATH

# FIXME:  we should make HARP debian packages to speed this up.
wget https://github.com/tskisner/HARP/releases/download/v${HARP_VERSION}/harp-${HARP_VERSION}.tar.gz
tar xzvf harp-${HARP_VERSION}.tar.gz
cd harp-${HARP_VERSION}
./configure --prefix=${HOME}/miniconda/envs/test --disable-mpi --disable-python && make && make install
cd ..

wget https://github.com/desihub/specex/archive/v${SPECEX_VERSION}.tar.gz
tar xzvf v${SPECEX_VERSION}.tar.gz
cd specex-${SPECEX_VERSION}
SPECEX_PREFIX=${HOME}/miniconda/envs/test make install
cd ..
