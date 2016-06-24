.. _intro:


Overview
==============

The DESI spectroscopic pipeline is a collection of software designed to 
efficiently take DESI raw data and produce redshift estimates.  This software
consists of low-level functions that perform individual processing tasks as 
well as higher-level tools to collectively run whole steps of the pipeline.


.. _install:

Installation and Requirements
================================

This software requires and interfaces with other external software packages.
This document assumes that you are setting up a "development" software stack 
from scratch based on the latest versions of the DESI tools.


External Dependencies
------------------------

Through whatever means, ensure that you have a recent python 2.7.x software
stack (e.g. Anaconda).  You must also check that you have installed through
your package manager or manually:

    * BLAS / LAPACK (OpenBLAS, Accelerate framework, etc)
    * CFITSIO

Use conda, pip or your package manager to install:

    * requests
    * astropy
    * fitsio
    * pyyaml
    * speclite

If you wish to run the desispec pipeline on a cluster, also ensure that you
have installed mpi4py (which obviously requires a working MPI installation).


DESI Affiliated Dependencies
---------------------------------

Next we will install dependencies associated with DESI.  We will be installing
these tools from source to a single location, and will use this for active
development.  We begin by deciding where to install everything



