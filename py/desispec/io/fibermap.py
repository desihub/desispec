"""
desispec.io.fibermap
====================

IO routines for fibermap.
"""
import os
import warnings
import numpy as np
from astropy.io import fits

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
    ('POSITIONER', 'i4'),
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
    POSITIONER   = "Positioner ID [0-4999]",
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
    MAG          = "magitude",
    FILTER       = "SDSS_R, DECAM_Z, WISE1, etc."
)

def empty_fibermap(nspec, specmin=0):
    """Return an empty fibermap ndarray to be filled in.
    """
    fibermap = np.zeros(nspec, dtype=fibermap_columns)
    fibermap['FIBER'] = np.arange(specmin, specmin+nspec)
    fibers_per_spectrograph = 500
    fibermap['SPECTROID'] = fibermap['FIBER'] // fibers_per_spectrograph
    return fibermap

def write_fibermap(outfile, fibermap, header=None):
    """Write fibermap binary table to outfile.

    Args:
        outfile (str): output filename
        fibermap: ndarray with named columns of fibermap data
        header: header data to include in same HDU as fibermap

    Returns:
        write_fibermap (str): full path to filename of fibermap file written.
    """
    outfile = makepath(outfile)

    #- astropy.io.fits incorrectly generates warning about 2D arrays of strings
    #- Temporarily turn off warnings to avoid this; desispec.test.test_io will
    #- catch it if the arrays actually are written incorrectly.
    hdr = fitsheader(header)
    add_dependencies(hdr)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_bintable(outfile, fibermap, hdr, comments=fibermap_comments,
            extname="FIBERMAP", clobber=True)

    return outfile


def read_fibermap(filename, header=False) :
    """Reads a fibermap file and returns its data as a numpy structured array
    
    Args:
        filename : input file name
        
    Options:
        header : if True, return (fibermap, header) tuple
    """

    if not os.path.isfile(filename) :
        raise IOError("cannot open"+filename)

    fibermap, hdr = fits.getdata(filename, 'FIBERMAP', header=True)
    fibermap = np.asarray(fibermap)

    if header:
        return fibermap, hdr
    else:
        return fibermap
