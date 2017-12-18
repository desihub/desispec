"""
desispec.io.fibermap
====================

IO routines for fibermap.
"""
import os
import warnings
import numpy as np
from astropy.table import Table

from desiutil.depend import add_dependencies
from desispec.io.util import fitsheader, write_bintable, makepath

fibermap_columns = [
    ('OBJTYPE', (str, 10)),
    ('TARGETCAT', (str, 20)),
    ('BRICKNAME', (str, 8)),
    ('TARGETID', 'i8'),
    ('DESI_TARGET', 'i8'),
    ('BGS_TARGET', 'i8'),
    ('MWS_TARGET', 'i8'),
    ('MAG', 'f4', (5,)),
    ('FILTER', (str, 10), (5,)),
    ('SPECTROID', 'i4'),
    ('POSITIONER', 'i4'),   #- deprecated; use LOCATION
    ('LOCATION',   'i4'),
    ('DEVICE_LOC', 'i4'),
    ('PETAL_LOC',  'i4'),
    ('FIBER', 'i4'),
    ('LAMBDAREF', 'f4'),
    ('RA_TARGET', 'f8'),
    ('DEC_TARGET', 'f8'),
    ('RA_OBS', 'f8'), ('DEC_OBS', 'f8'),
    ('X_TARGET', 'f8'), ('Y_TARGET', 'f8'),
    ('X_FVCOBS', 'f8'), ('Y_FVCOBS', 'f8'),
    ('Y_FVCERR', 'f4'), ('X_FVCERR', 'f4'),
    ]

fibermap_comments = dict(
    FIBER        = "Fiber ID [0-4999]",
    POSITIONER   = "Positioner ID [0-4999] (deprecated)",
    LOCATION     = "Positioner location ID 1000*PETAL + DEVICE",
    PETAL_LOC    = "Petal location on focal plane [0-9]",
    DEVICE_LOC   = "Device location on petal [0-542]",
    SPECTROID    = "Spectrograph ID [0-9]",
    TARGETID     = "Unique target ID",
    TARGETCAT    = "Name/version of the target catalog",
    BRICKNAME    = "Brickname from target imaging",
    OBJTYPE      = "Target type [ELG, LRG, QSO, STD, STAR, SKY]",
    LAMBDAREF    = "Reference wavelength at which to align fiber",
    DESI_TARGET  = "DESI dark+calib targeting bit mask",
    BGS_TARGET   = "DESI Bright Galaxy Survey targeting bit mask",
    MWS_TARGET   = "DESI Milky Way Survey targeting bit mask",
    RA_TARGET    = "Target right ascension [degrees]",
    DEC_TARGET   = "Target declination [degrees]",
    X_TARGET     = "X on focal plane derived from (RA,DEC)_TARGET",
    Y_TARGET     = "Y on focal plane derived from (RA,DEC)_TARGET",
    X_FVCOBS     = "X location observed by Fiber View Cam [mm]",
    Y_FVCOBS     = "Y location observed by Fiber View Cam [mm]",
    X_FVCERR     = "X location uncertainty from Fiber View Cam [mm]",
    Y_FVCERR     = "Y location uncertainty from Fiber View Cam [mm]",
    RA_OBS       = "RA of obs from (X,Y)_FVCOBS and optics [deg]",
    DEC_OBS      = "dec of obs from (X,Y)_FVCOBS and optics [deg]",
    MAG          = "magnitudes in each of the filters",
    FILTER       = "SDSS_R, DECAM_Z, WISE1, etc.",
    #- Optional columns, used by spectra but not by frames
    NIGHT        = "Night of exposure YYYYMMDD",
    EXPID        = "Exposure ID",
    TILEID       = "Tile ID",
)

def empty_fibermap(nspec, specmin=0):
    """Return an empty fibermap ndarray to be filled in.

    Args:
        nspec: (int) number of fibers(spectra) to include

    Options:
        specmin: (int) starting spectrum index
    """
    import desimodel.io

    assert 0 <= nspec <= 5000, "nspec {} should be within 0-5000".format(nspec)
    fibermap = Table(np.zeros(nspec, dtype=fibermap_columns))
    fibermap['FIBER'] = np.arange(specmin, specmin+nspec)
    fibers_per_spectrograph = 500
    fibermap['SPECTROID'] = fibermap['FIBER'] // fibers_per_spectrograph

    fiberpos = desimodel.io.load_fiberpos()
    ii = slice(specmin, specmin+nspec)
    fibermap['X_TARGET'][:]   = fiberpos['X'][ii]
    fibermap['Y_TARGET'][:]   = fiberpos['Y'][ii]
    fibermap['X_FVCOBS'][:]   = fiberpos['X'][ii]
    fibermap['Y_FVCOBS'][:]   = fiberpos['Y'][ii]
    fibermap['POSITIONER'][:] = fiberpos['LOCATION'][ii]   #- deprecated
    fibermap['LOCATION'][:]   = fiberpos['LOCATION'][ii]
    fibermap['PETAL_LOC'][:]  = fiberpos['PETAL'][ii]
    fibermap['DEVICE_LOC'][:] = fiberpos['DEVICE'][ii]
    fibermap['LAMBDAREF'][:]  = 5400.0

    assert set(fibermap.keys()) == set([x[0] for x in fibermap_columns])
        
    return fibermap

def write_fibermap(outfile, fibermap, header=None):
    """Write fibermap binary table to outfile.

    Args:
        outfile (str): output filename
        fibermap: astropy Table of fibermap data
        header: header data to include in same HDU as fibermap

    Returns:
        write_fibermap (str): full path to filename of fibermap file written.
    """
    outfile = makepath(outfile)

    #- astropy.io.fits incorrectly generates warning about 2D arrays of strings
    #- Temporarily turn off warnings to avoid this; desispec.test.test_io will
    #- catch it if the arrays actually are written incorrectly.
    if header is not None:
        hdr = fitsheader(header)
    else:
        hdr = fitsheader(fibermap.meta)

    add_dependencies(hdr)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_bintable(outfile, fibermap, hdr, comments=fibermap_comments,
            extname="FIBERMAP", clobber=True)

    return outfile


def read_fibermap(filename) :
    """Reads a fibermap file and returns its data as an astropy Table
    
    Args:
        filename : input file name
    """
    #- Implementation note: wrapping Table.read() with this function allows us
    #- to update the underlying format, extension name, etc. without having
    #- to change every place that reads a fibermap.
    
    return Table.read(filename, 'FIBERMAP')
