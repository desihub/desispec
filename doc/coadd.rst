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

Inspect a brick file in iPython using, e.g.::

    import os,os.path
    import astropy.io.fits as fits
    from astropy.table import Table
    brick = fits.open(os.path.join(os.getenv('DESI_SPECTRO_REDUX'),os.getenv('PRODNAME'),'bricks','3582m005','brick-r-3582m005.fits'))
    info = Table.read(brick,hdu=4)
    print info
    plt.errorbar(x=brick[2].data,y=brick[0].data[0],yerr=brick[1].data[0]**-0.5)

Notes
-----

* The brick filenames have the format `brick-{band}-{expid}.fits`, where `band` is one of [rbz], which differs from the current data model (which is missing the `{band}`).
* Bricks contain a single wavelength grid in HDU2, the same as current CFRAMES, but different from the CFRAME data model (where HDU2 is a per-object mask).
* The order of objects appearing in brick HDUs 0-3 (which are copied from the corresponding CFRAMEs) matches the order of rows in HDU4 (which are copied from the corresponding FIBERMAP).
* HDU4 adds NIGHT and EXPID columns, to distinguish repeat observations of the same object.
* The NIGHT column in HDU4 has type i4, not string. Is this a problem?
* The 5*S10 FILTER values in the FIBERMAP are combined into a single comma-separated list stored as a single S50 FILTER value in HDU4 of the brick file.  This is a workaround until we sort out issues with astropy.io.fits and cfitsio handling of 5*S10 arrays.
