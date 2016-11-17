.. _install:


Installation
===============

The DESI spectroscopic pipeline requires and interfaces with other external software packages.  This document assumes that you are setting up a "development" software stack from scratch based on the latest versions of the DESI tools.


External Dependencies
------------------------

In order to set up a working DESI pipeline, there are several external software packages that must be installed on your system.  There are several ways of obtaining these dependencies:  using your OS package manager, using a third-party package manager (e.g. macports, etc), or using one of the conda-based Python distributions (e.g. Anaconda from Continuum Analytics).

The list of general software outside the DESI project which is needed for the spectroscopic pipeline is:

    * BLAS / LAPACK (OpenBLAS, MKL, Accelerate framework, etc)
    * CFITSIO
    * BOOST
    * requests
    * matplotlib
    * scipy
    * astropy
    * fitsio
    * pyyaml
    * speclite

If you wish to run the pipeline on a cluster, also ensure that you have installed mpi4py (which obviously requires a working MPI installation).

Installing Dependencies on a Linux Workstation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For development, debugging, and small tests it is convenient to install the pipeline tools on a laptop or workstation.  If you are using a Linux machine, you can get all the dependencies (except fitsio and speclite, which are pip-installable) from your distribution's package manager.  Alternatively, you can install just the non-python dependencies with your package manager and then use Anaconda for your python stack.  On OS X, I recommend using macports to get at least the non-python dependencies, and perhaps all of the python tools as well.

**Example:  Ubuntu GNU/Linux**

On an Ubuntu machine, you could install all the dependencies with::

    %> sudo apt-get install libboost-all-dev libcfitsio-dev \
        libopenblas-dev liblapack-dev python3-matplotlib \
        python3-scipy python3-astropy python3-requests \
        python3-yaml python3-mpi4py

    %> pip install --no-binary :all: fitsio speclite iniparser


Installing Dependencies on an OS X System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installing scientific software on OS X is often more difficult than Linux, since Apple is primarily concerned with development of apps using Xcode.  The approach described here for installing desispec dependencies seems to get the job done with the fewest steps.

First, install homebrew::

    %> /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Now use homebrew to install CFITSIO, BOOST, and OpenMPI::

    %> brew install cfitsio
    %> brew install boost
    %> brew install openmpi

We are going to use homebrew to install our python stack.  Some people prefer Anaconda, but that distribution has several problems on OS X.  When installing python, we add the flag to indicate we want the python3 versions as well::

    %> brew install homebrew/python/mpi4py --with-python3
    %> brew install homebrew/python/scipy --with-python3
    %> brew install homebrew/python/matplotlib --with-python3

The rest of the dependencies we can install with pip::

    %> pip3 install requests pyyaml iniparser speclite astropy
    %> pip3 install --no-binary :all: fitsio


Dependencies at NERSC
~~~~~~~~~~~~~~~~~~~~~~~~~

At NERSC there is already a conda-based python stack and a version of the non-python dependencies installed.  You can add the necessary module search path by doing (you can safely add this to your ~/.bashrc.ext)::

    %> module use /global/common/${NERSC_HOST}/contrib/desi/modulefiles

and then whenever you want to load the software::

    %> module load desi-conda



DESI Affiliated Dependencies
---------------------------------

Now we should have our base software stack set up and next we will install dependencies associated with DESI.  For this example, we will install DESI software to our home directory.  On NERSC systems, the $HOME directory is not as performant as $SCRATCH in terms of python startup time.  However, installing software to $SCRATCH (and the necessary steps to prevent it from being purged) is beyond the scope of this document.  


Create a Shell Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a bash function that we will use to load our installed desi software into our environment::

    desidev () {
        # This is the install location of our desi software.
        # If you are not at NERSC, then change this to something
        # without "NERSC_HOST" in the name.
        desisoft="${HOME}/desi-${NERSC_HOST}"
        
        # Set environment variables
        export CPATH=${desisoft}/include:${CPATH}
        export LIBRARY_PATH=${desisoft}/lib:${LIBRARY_PATH}
        export LD_LIBRARY_PATH=${desisoft}/lib:${LD_LIBRARY_PATH}
        export PYTHONPATH=${desisoft}/lib/python3.5/site-packages:${PYTHONPATH}

        # Special setup for redmonster
        red="${HOME}/git-${NERSC_HOST}/redmonster"
        export PYTHONPATH=${red}/python:${PYTHONPATH}
        export REDMONSTER_TEMPLATES_DIR=${red}/templates

        # Choose what data files to use- these locations
        # are for NERSC.
        export DESI_ROOT=/project/projectdirs/desi
        export DESIMODEL=${DESI_ROOT}/software/edison/desimodel/master
        export DESI_BASIS_TEMPLATES=${DESI_ROOT}/spectro/templates/basis_templates/v2.2
        export STD_TEMPLATES=${DESI_ROOT}/spectro/templates/star_templates/v1.1/star_templates_v1.1.fits
    }

Now log out and back in.  We should pre-create the python package directory the first time we install things::

    %> mkdir -p ${HOME}/desi-${NERSC_HOST}/lib/python3.5/site-packages

At NERSC, first load our dependencies::

    %> module load desi-conda

and then execute our shell function::

    %> desidev


Install Release Tarballs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we are ready to install software to this location.  Some packages (HARP), do not currently change rapidly and we can just install them from a released tarball.  If you are building on a workstation or laptop, download the latest release of HARP from https://github.com/tskisner/HARP/releases and install::

    %> cd harp-1.0.1
    %> ./configure --disable-python --disable-mpi \
       --prefix="${HOME}/desi-${NERSC_HOST}"

.. NOTE::

    At NERSC, the HARP package is already included in the software loaded by the desi-conda module.  You do not need to build it at NERSC.


Organize Your Git Clones
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the purposes of this document, we assume that all DESI git clones reside in $HOME/git-$NERSC_HOST.  You will need to get the following repos.  Some of these are not strictly necessary for the spectroscopic pipeline, but are useful for simulating data as part of the integration tests::

    %> cd $HOME/git-$NERSC_HOST
    %> git clone git@github.com:desihub/desiutil.git
    %> git clone git@github.com:desihub/desimodel.git
    %> git clone git@github.com:desihub/desitarget.git
    %> git clone git@github.com:desihub/desisim.git
    %> git clone git@github.com:desihub/specter.git
    %> git clone git@github.com:desihub/specex.git
    %> git clone git@github.com:desihub/desispec.git
    %> git clone git@github.com:desihub/redmonster.git

Now we are ready to install the various DESI packages from their git source trees.  Let's go into our git directory and create a small helper script which will update your install any time you update your source trees::

    %> cd $HOME/git-$NERSC_HOST
    %> cat install.sh
    
    #!/bin/bash

    # This should be your actual install location...
    pref="${HOME}/desi-${NERSC_HOST}"

    cd specex
    make clean
    SPECEX_PREFIX=${pref} make -j 4 install
    cd ..

    for pkg in desiutil desimodel desitarget desisim specter desispec; do
        cd ${pkg}
        python setup.py clean
        python setup.py install --prefix=${pref}
        cd ..
    done

For the initial install, and any you update your source tree versions, do (make sure the install.sh script is executable)::

    %> ./install.sh

Now your DESI software stack is complete.  Just run the "desidev" shell function to load everything into your environment, and rerun the install.sh script any time you update your source versions.


