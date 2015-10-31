#!/bin/bash -x

# CONDA
conda create --yes -n test -c astropy-ci-extras python=$PYTHON_VERSION pip
source activate test

# EGG_INFO
if [[ $SETUP_CMD == egg_info ]]
then
  exit  # no more dependencies needed
fi

# PEP8
if [[ $MAIN_CMD == pep8* ]]
then
  $PIP_INSTALL pep8
  exit  # no more dependencies needed
fi

# CORE DEPENDENCIES
conda install --yes pytest Cython jinja2 psutil pyyaml requests

# NUMPY scipy
conda install --yes numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION matplotlib

# ASTROPY
if [[ $ASTROPY_VERSION == development ]]
then
  $PIP_INSTALL git+http://github.com/astropy/astropy.git#egg=astropy
  export CONDA_INSTALL="conda install --yes numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION"
else
  conda install --yes astropy=$ASTROPY_VERSION
  export CONDA_INSTALL="conda install --yes numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION astropy=$ASTROPY_VERSION"
fi

# Now set up shortcut to conda install command to make sure the Python and Numpy
# versions are always explicitly specified.

# OPTIONAL DEPENDENCIES
if $OPTIONAL_DEPS
then
  $CONDA_INSTALL h5py scikit-image pandas
  $PIP_INSTALL beautifulsoup4
fi

# DESI DEPENDENCIES
$PIP_INSTALL git+https://github.com/desihub/desiutil.git@${DESIUTIL_VERSION}#egg=desiutil

# DOCUMENTATION DEPENDENCIES
# build_sphinx needs sphinx and matplotlib (for plot_directive). Note that
# this matplotlib will *not* work with py 3.x, but our sphinx build is
# currently 2.7, so that's fine
if [[ $SETUP_CMD == build_sphinx* ]]
then
  $CONDA_INSTALL Sphinx=$SPHINX_VERSION Pygments matplotlib
fi

# COVERAGE DEPENDENCIES
# cpp-coveralls must be installed first.  It installs two identical
# scripts: 'cpp-coveralls' and 'coveralls'.  The latter will overwrite
# the script installed by 'coveralls', unless it's installed first.
if [[ $SETUP_CMD == 'test --coverage' ]]
then
  $PIP_INSTALL cpp-coveralls;
  $PIP_INSTALL coverage coveralls;
fi
