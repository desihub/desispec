"""
io routines for fibermap

"""
import os
from astropy.io import fits

def read_fibermap(filename) :
    """
    reads a fibermap fits file and returns its data
    """
    
    if not os.path.isfile(filename) :
        raise IOError("cannot open"+filename)
    hdulist = fits.open(filename)
    tbdata = hdulist[1].data
    
    return tbdata
