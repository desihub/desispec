"""
io routines for flux calibration

"""
import os
from astropy.io import fits


# this is really temporary
# the idea is to have a datamodel for calibration stars spectra
def read_stellar_models(filename) :

    """
    read stellar models
    """
    flux=fits.getdata(filename, 0).astype('float64')
    wave=fits.getdata(filename, 1).astype('float64')
    fibers=fits.getdata(filename, 2).astype(int)
    return flux,wave,fibers

