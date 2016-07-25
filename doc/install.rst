.. _install:


Installation
===============

The DESI spectroscopic pipeline requires and interfaces with other external software packages.  This document assumes that you are setting up a "development" software stack from scratch based on the latest versions of the DESI tools.


External Dependencies
------------------------

In order to set up a working DESI pipeline, there are several external software packages that must be installed on your system.  There are several ways of obtaining these dependencies:  using your OS package manager, using a third-party package manager (e.g. macports, hpcports, etc), or using one of the conda-based Python distributions (e.g. Anaconda from Continuum Analytics).

The list of general software outside the DESI project which is needed for the spectroscopic pipeline is:

    * BLAS / LAPACK (OpenBLAS, MKL, Accelerate framework, etc)
    * CFITSIO
    * BOOST
    * requests
    * matplotlib
    * astropy
    * fitsio
    * pyyaml
    * speclite

If you wish to run the pipeline on a cluster, also ensure that you have installed mpi4py (which obviously requires a working MPI installation).

For this documentation, we are going to assume that you will be setting up all dependencies from scratch using Anaconda.  We will document the differences at each step when installing at NERSC or on a simple workstation.  Note that for NERSC systems, the DESI project will likely soon provide a ready-made conda environment for this base software stack.


Create a Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using a workstation, download and install Anaconda.  After that, you should have the "conda" command in your $PATH.  

.. topic:: At NERSC

    If you are using a Cray system at NERSC, then do::

        %> module swap PrgEnv-intel PrgEnv-gnu
        %> module load python/2.7-anaconda

    to get conda into your $PATH.

.. topic:: At NERSC

    To avoid long pathnames in the conda environment label, do some **one-time** commands to set up the location of your conda environments::

        %> mkdir $HOME/conda-envs
        %> conda config --add envs_dirs $HOME/conda-envs
        %> conda config --add channels defaults
        %> conda config --set show_channel_urls yes


Now create a conda environment and activate it.  We will call this environment "desi" in the case of a workstation install, or "desi-$NERSC_HOST" in the case of installing at NERSC.  If you are installing on a workstation, you can include mpi4py in this create step::

    %> conda create -n desi numpy scipy astropy matplotlib requests \
       yaml pyyaml autoconf automake libtool boost mpi4py

.. topic:: At NERSC

    Here we name the conda env so that it includes the machine name.  Also, we **DO NOT** install mpi4py with conda::

        %> conda create -n desi-$NERSC_HOST numpy scipy astropy \
           matplotlib requests yaml pyyaml autoconf automake libtool \
           boost

And now activate the environment::

    %> source activate desi

.. topic:: At NERSC

    We must specify the machine specific name::

        %> source activate desi-$NERSC_HOST

.. topic:: At NERSC

    mpi4py installation requires special care at NERSC, since it must be built using the Cray compiler wrappers.  Here is how to install mpi4py manually at NERSC::

        %> wget https://pypi.python.org/packages/ee/b8/f443e1de0b6495479fc73c5863b7b5272a4ece5122e3589db6cd3bb57eeb/mpi4py-2.0.0.tar.gz#md5=4f7d8126d7367c239fd67615680990e3
        %> tar xzvf mpi4py-2.0.0.tar.gz
        %> cd mpi4py-2.0.0
        %> python setup.py build --mpicc=cc --mpicxx=CC
        %> python setup.py install

Finally, we pip install a couple packages that are not available through conda::

    %> pip install fitsio
    %> pip install speclite


DESI Affiliated Dependencies
---------------------------------

Now we should have our base software stack set up and next we will install dependencies associated with DESI.  For simplicity, we will be installing this software directly into the conda environment we created in the last section.

.. warning::

    On NERSC machines, a conda environment contains a full TCL/TK installation in the lib subdirectory.  We will be installing shared libraries to this location and adding that directory to $LD_LIBRARY_PATH.  As soon as these conda-specific TCL libraries are in $LD_LIBRARY_PATH, the environment modules will cease to work.  Do not attempt to use module commands after running the shell function below.

For this example, we will install DESI software to our conda environment, which we located in $HOME.  On NERSC systems, the $HOME directory is not as performant as $SCRATCH in terms of python startup time.  However, installing software to $SCRATCH (and the necessary steps to prevent it from being purged) is beyond the scope of this document.  

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

Create a Shell Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a bash function that we will use to load our installed desi software into our environment.  This will also load our base software stack from the previous section::

    desi () {
        # At NERSC, we need these commands first
        module swap PrgEnv-intel PrgEnv-gnu
        module load python/2.7-anaconda
        
        # Activate the conda env, which adds the bin directory to PATH
        source activate desi-${NERSC_HOST}
        
        # This is the install location of our desi software
        desisoft="${HOME}/conda-envs/desi-${NERSC_HOST}"
        
        # Set environment variables
        export CPATH=${desisoft}/include:${CPATH}
        export LIBRARY_PATH=${desisoft}/lib:${LIBRARY_PATH}
        export LD_LIBRARY_PATH=${desisoft}/lib:${LD_LIBRARY_PATH}
        export PYTHONPATH=${desisoft}/lib/python2.7/site-packages:${PYTHONPATH}

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

Now log out and back in, and then execute our shell function::

    %> desi

Now we are ready to install software to this location.  Although it is not technically "DESI" software, we need to install CFITSIO manually since it is not available through conda::

    %> wget http://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio3390.tar.gz
    %> tar xzvf cfitsio3390.tar.gz
    %> cd cfitsio
    %> ./configure --prefix=$HOME/conda-envs/desi-$NERSC_HOST
    %> make
    %> make shared
    %> make install

(if not at NERSC, change "desi-$NERSC_HOST" to "desi" above).  Next we install the HARP tools, but only the minimal set needed by SPECEX::

    %> wget https://github.com/tskisner/HARP/releases/download/v1.0.1/harp-1.0.1.tar.gz
    %> tar xzvf harp-1.0.1.tar.gz
    %> cd harp-1.0.1
    %> ./configure --disable-python --disable-mpi \
        --with-blas="-lmkl_rt -fopenmp -lpthread -lm -ldl" \
        --prefix=$HOME/conda-envs/desi-$NERSC_HOST
    %> make install

.. NOTE::

    The wget version at NERSC may be too old to fetch the harp tarball above.  You may have to download it on another computer and copy it to NERSC.

Now we are ready to install the various DESI packages.  Let's go into our git directory and create a small helper script which will update your install any time you update your source trees::

    %> cd $HOME/git-$NERSC_HOST
    %> cat install.sh
    
    #!/bin/bash
    pref="${HOME}/conda-envs/desi-${NERSC_HOST}"

    cd specex
    make clean
    SPECEX_PREFIX=${pref} make install
    cd ..

    for pkg in desiutil desimodel desitarget desisim specter desispec; do
        cd ${pkg}
        python setup.py clean
        python setup.py develop
        cd ..
    done


For the initial install, and any you update your source tree versions, do (make sure the install.sh script is executable)::

    %> ./install.sh

Now your DESI software stack is complete.  Just run the "desi" shell function to load everything into your environment, and rerun the install.sh script any time you update your source versions.








