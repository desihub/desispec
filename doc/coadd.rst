=======================================
Coaddition of Spectroperfect Reductions
=======================================

This document covers coadd implementation details and mock-data tests.
For details on the coadd dataflow and algorithms used for combining spectra,
refer to `DESI-doc-1056 <https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=1056>`_.

Implementation
++++++++++++++

Files
~~~~~

All spectra are grouped by brick. There are three types of brick file under ``$DESI_SPECTRO_REDUX/$PRODNAME//bricks/{brickid}/``:

* The brick files contains all exposures of every target that has been observed in the brick, by band.
* The band coadd files contain the coadd of all exposures for each target, by band.
* The global coadd files contain the coadd of band coadds for each target.

All files have the same structure with 4 HDUs:

* HDU0: Flux vectors for each spectrum.
* HDU1: Ivar vectors for each spectrum.
* HDU2: The common wavelength grid used for all spectra.
* HDU4: Binary table of metadata.

See the relevant `data model descriptions
<https://desi.lbl.gov/trac/browser/code/desiDataModel/trunk/doc/DESI_SPECTRO_REDUX/PRODNAME/bricks/BRICKID>`_
for details (these are not in synch with the mock data challenge files as of 23-Mar-2015).

Wavelength Grids
~~~~~~~~~~~~~~~~

All brick files have their wavelength grid in HDU2, as summarized in the table
below. Note that we do not use a log-lambda grid for the global coadd across
bands, since this would not be a good match to the wavelength resolution at
the red ends of r and z cameras. See the note for details.

===== ======= ======= ======= ======= ==================
Band  Min(A)  Max(A)  Nbins   Size(A) Files
===== ======= ======= ======= ======= ==================
b     3579.0  5938.8  3934    0.6     brick, band coadd
r     5635.0  7730.8  3494    0.6     brick, band coadd
z     7445.0  9824.0  3966    0.6     brick, band coadd
all   3579.0  9825.0  6247    1.0     global coadd
===== ======= ======= ======= ======= ==================

Metadata
~~~~~~~~

The brick file and two co-add files all have a table with the same format in
HDU4, with one entry per exposure of each object observed in the brick. Most
of its columns are copied directly from the exposure fibermaps, except for
the last three columns which identify the exposure and offset of the object's
spectrum in the other HDUs.  The table columns are listed below.

============ ======================================================
Name         Description
============ ======================================================
FIBER        Fiber ID [0-4999]
POSITIONER   Positioner ID [0-4999]
SPECTROID    Spectrograph ID [0-9]
TARGETID     Unique target ID
TARGETCAT    Name/version of the target catalog
OBJTYPE      Target type [ELG, LRG, QSO, STD, STAR, SKY]
LAMBDAREF    Reference wavelength at which to align fiber
TARGET_MASK0 Targeting bit mask
RA_TARGET    Target right ascension [degrees]
DEC_TARGET   Target declination [degrees]
X_TARGET     X on focal plane derived from (RA,DEC)_TARGET
Y_TARGET     Y on focal plane derived from (RA,DEC)_TARGET
X_FVCOBS     X location observed by Fiber View Cam [mm]
Y_FVCOBS     Y location observed by Fiber View Cam [mm]
X_FVCERR     X location uncertainty from Fiber View Cam [mm]
Y_FVCERR     Y location uncertainty from Fiber View Cam [mm]
RA_OBS       RA of obs from (X,Y)_FVCOBS and optics [deg]
DEC_OBS      dec of obs from (X,Y)_FVCOBS and optics [deg]
MAG          magitude
FILTER       SDSS_R, DECAM_Z, WISE1, etc.
NIGHT        Date string for the night of observation YYYYMMDD
EXPID        Integer exposure number
INDEX        Index of this object in other HDUs
============ ======================================================

Programs
~~~~~~~~

The following programs are used to implement the coadd part of the pipeline:

* `desi_make_bricks`: Create brick files from all exposures taken in one night. Reads exposures from cframe files and adds metadata from the exposure fibermap.
* `desi_update_coadds`: Update the coadds for a single brick. Reads exposures from brick files and writes the corresonding band coadd and global coadd files.

An additional program `desi_inspect` displays the information and creates a plot summarizing the coadd results for a single target.

Benchmarks
~~~~~~~~~~

The rate-limiting step for performing coadds is the final conversion from `Cinv` and `Cinv_f` to `flux`, `ivar` and `resolution` in :meth:`desispec.coaddition.Spectrum.finalize`.  The computation time is dominated by one operation: solving the eigenvalue program for a large symmetric real-valued matrix using :func:`scipy.linalg.eigh`. The computation also involves inverting a real-valued resolution matrix using :func:`scipy.linalg.inv`, but this is relatively fast.

For a typical r-band coadd on a 2014 MacBook Pro, the total time for a single target is about 12 seconds, dominated by `eigh` (10.6s) and `inv` (1.0s). The time for a global coadd will be longer because of the larger matrices involved.

Interactive tests run on edison@nesrc indicate that it takes about 20s for each single-band coadd and 90s for the global coadd, for a total of about 150s per target.  Note that the coadd step can be parallelized across bricks to reduce the wall-clock time required to process an exposure.  The biggest speed improvement would likely come from using a sparse matrix eigensolver, or adjusting the algorithm to be able to use an incomplete set of eigenmodes (the :func:`scipy.sparse.linalg.eigsh` function can not calculate the full spectrum of eigenmodes).

Notes
~~~~~

* The brick filenames have the format ``brick-{band}-{expid}.fits``, where ``band`` is one of [rbz], which differs from the current data model (which is missing the ``{band}``).
* Bricks contain a single wavelength grid in HDU2, the same as current CFRAMES, but different from the CFRAME data model (where HDU2 is a per-object mask).
* The NIGHT column in HDU4 has type i4, not string. Is this a problem?
* The 5*S10 FILTER values in the FIBERMAP are combined into a single comma-separated list stored as a single S50 FILTER value in HDU4 of the brick file.  This is a workaround until we sort out issues with astropy.io.fits and cfitsio handling of 5*S10 arrays.
* The mock resolution matrices do not have np.sum(R,axis=1) == 1 for all rows and go slightly negative in the tails.
* The wlen values in HDU2 have some roundoff errors, e.g., z-band wlen[-1] = 9824.0000000014425
* Masking via ivar=0 is implemented but not well tested yet.
* We need a way to programmatically determine the brick name given a target ID, in order to locate the relevant files. Otherwise, target ID is not a useful way to define a sample (a la plate-mjd-fiber or ThingID) and an alternative is needed for downstream science users.
* The global coadd sometimes find negative eigenvalues for Cinv or a singular R.T. These cases need to be investigated.

Mock Data Tests
+++++++++++++++

.. warning::
    The description of environment setup and installation below may be out of date.

DESI Environment
~~~~~~~~~~~~~~~~

Ssh to cori.nersc.gov (remember to use `ssh -A` to propagate your keys for github access) and::

    source /project/projectdirs/desi/software/desi_environment.sh

Installation
~~~~~~~~~~~~

Clone the git package and select the co-add development branch (which should soon be merged into the master branch, making the last command unecessary)::

    git clone git@github.com:desihub/desispec.git
    cd desispec
    git checkout \#6

Per-Login Setup
~~~~~~~~~~~~~~~

Manually set paths for using this installation (assuming `bash`)::

    cd desispec
    export PATH=$PWD/bin:$PATH
    export PYTHONPATH=$PWD/py:$PYTHONPATH

Set pipeline paths::

    export DESI_SPECTRO_REDUX=$DESI_ROOT/spectro/redux
    export PRODNAME=sjb/cedar2a
    export DESI_SPECTRO_SIM=$DESI_ROOT/spectro/sim
    export DESI_SPECTRO_DATA=$DESI_SPECTRO_SIM/alpha-5

Run Tests
~~~~~~~~~

Convert mocks cframes and fibermaps into brick files using::

    rm -rf $DESI_SPECTRO_REDUX/$PRODNAME/bricks
    desi_make_bricks.py --night 20150211 --verbose

Note that the code is not yet smart enough to do the right thing for exposures that have already been added to brick files, hence the `rm` command above.

Update coadds for a single brick::

    rm -rf $DESI_SPECTRO_REDUX/$PRODNAME/bricks/3587m010/coadd*
    desi_update_coadds.py --brick 3587m010 --verbose

Look at a single target in this brick (this is an LRG)::

    desi_inspect.py --brick 3587m010 --target 3640213155238558158 --verbose

Inspect a brick file in iPython using, e.g.::

    import os,os.path
    import astropy.io.fits as fits
    from astropy.table import Table
    brick = fits.open(os.path.join(os.getenv('DESI_SPECTRO_REDUX'),os.getenv('PRODNAME'),'bricks','3587m010','brick-r-3587m010.fits'))
    info = Table.read(brick,hdu=4)
    print info
    plt.errorbar(x=brick[2].data,y=brick[0].data[0],yerr=brick[1].data[0]**-0.5)

Run unit tests::

    python -m desispec.resolution
