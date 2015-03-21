Coaddition Notes
================

DESI Environment
----------------

Ssh to edison.nersc.gov (remember to use `ssh -A` to propagate your keys for github access) and::

    source /project/projectdirs/desi/software/modules/desi_environment.sh

Installation
------------

Clone the git package and select the co-add development branch::

    git clone git@github.com:desihub/desispec.git
    cd desispec
    git checkout \#3

Per-Login Setup
---------------

Manually set paths for using this installation (assuming `bash`)::

    cd desispec
    export PATH=$PWD/bin:$PATH
    export PYTHONPATH=$PWD/py:$PYTHONPATH

Set pipeline paths::

    export DESI_SPECTRO_REDUX=$DESI_ROOT/spectro/redux
    export PRODNAME=sjb/cedar2a
    export DESI_SPECTRO_SIM=$DESI_ROOT/spectro/sim
    export DESI_SPECTRO_DATA=$DESI_SPECTRO_SIM/alpha-5

Tests
-----

Convert mocks cframes and fibermaps into brick files using::

    rm -rf $DESI_SPECTRO_REDUX/$PRODNAME/bricks
    desi_make_bricks.py --night 20150211 --verbose

Note that the code is not yet smart enough to do the right thing for exposures that have already been added to brick files, hence the `rm` command above.

Update coadds for a single brick::

    rm -rf $DESI_SPECTRO_REDUX/$PRODNAME/bricks/3582p000/coadd*
    desi_update_coadds.py --brick 3582p000 --verbose

Look at a single target in this brick::

    desi_inspect.py --brick 3582p000 --id 7374379192747158494 --verbose

Inspect a brick file in iPython using, e.g.::

    import os,os.path
    import astropy.io.fits as fits
    from astropy.table import Table
    brick = fits.open(os.path.join(os.getenv('DESI_SPECTRO_REDUX'),os.getenv('PRODNAME'),'bricks','3582p000','brick-r-3582p000.fits'))
    info = Table.read(brick,hdu=4)
    print info
    plt.errorbar(x=brick[2].data,y=brick[0].data[0],yerr=brick[1].data[0]**-0.5)

Wavelength Grids
----------------

All brick files have their wavelength grid in HDU2, as summarized in the table below. Note that we do not use a log-lambda grid for the global coadd across bands, since this would not be a good match to the wavelength resolution at the red ends of r and z cameras. See the note for details.

===== ======= ======= ======= =======
Band  Min(A)  Max(A)  Nbins   Size(A)
===== ======= ======= ======= =======
b     3579.0  5938.8  3934    0.6
r     5635.0  7730.8  3494    0.6
z     7445.0  9824.0  3966    0.6
all   3579.0  9825.0  6247    1.0
===== ======= ======= ======= =======

Co-Add Table
------------

The brick file and two co-add files all have a table with the same format in HDU4, with one entry per exposure of each object observed in the brick. Most of its columns are copied directly from the exposure fibermaps, except for the last three columns which identify the exposure and offset of the object's spectrum in the other HDUs.  The table columns are listed below.

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

Notes
-----

* The brick filenames have the format `brick-{band}-{expid}.fits`, where `band` is one of [rbz], which differs from the current data model (which is missing the `{band}`).
* Bricks contain a single wavelength grid in HDU2, the same as current CFRAMES, but different from the CFRAME data model (where HDU2 is a per-object mask).
* The order of objects appearing in brick HDUs 0-3 (which are copied from the corresponding CFRAMEs) matches the order of rows in HDU4 (which are copied from the corresponding FIBERMAP).
* HDU4 adds NIGHT and EXPID columns, to distinguish repeat observations of the same object.
* The NIGHT column in HDU4 has type i4, not string. Is this a problem?
* The 5*S10 FILTER values in the FIBERMAP are combined into a single comma-separated list stored as a single S50 FILTER value in HDU4 of the brick file.  This is a workaround until we sort out issues with astropy.io.fits and cfitsio handling of 5*S10 arrays.
* The mock resolution matrices do not have np.sum(R,axis=1) == 1 for all rows and go slightly negative in the tails.
* The wlen values in HDU2 have some roundoff errors, e.g., z-band wlen[-1] = 9824.0000000014425
