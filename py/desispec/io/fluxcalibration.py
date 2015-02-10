"""
io routines for flux calibration

"""
import os
from astropy.io import fits
from desispec.io.util import native_endian

# this is really temporary
# the idea is to have a datamodel for calibration stars spectra
def read_stellar_models(filename) :
    """
    read stellar models from filename
    
    returns flux[nspec, nwave], wave[nwave], fibers[nspec]
    """
    flux = native_endian(fits.getdata(filename, 0))
    wave = native_endian(fits.getdata(filename, 1))
    fibers = native_endian(fits.getdata(filename, 2))
    return flux,wave,fibers

