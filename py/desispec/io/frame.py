"""
desispec.io.frame
=================

I/O routines for Frame objects
"""
import os.path

import numpy as np
import scipy,scipy.sparse
from astropy.io import fits

from desispec.frame import Frame
from desispec.io import findfile
from desispec.io.util import fitsheader, native_endian, makepath
from desispec.log import get_logger

log = get_logger()

def write_frame(outfile, frame, header=None):
    """Write a frame fits file and returns path to file written.

    Args:
        outfile: full path to output file, or tuple (night, expid, channel)
        frame:  desispec.frame.Frame object with wave, flux, ivar...
        header: optional astropy.io.fits.Header or dict to override frame.header
        
    Returns:
        full filepath of output file that was written
        
    Note:
        to create a Frame object to pass into write_frame,
        frame = Frame(wave, flux, ivar, resolution_data)
    """
    outfile = makepath(outfile, 'frame')

    if header is not None:
        hdr = fitsheader(header)
    else:
        #hdr = fitsheader(frame.header)
        hdr = fitsheader(frame.meta)

    if 'SPECMIN' not in hdr:
        hdr['SPECMIN'] = 0
    if 'SPECMAX' not in hdr:
        hdr['SPECMAX'] = hdr['SPECMIN'] + frame.nspec

    hdus = fits.HDUList()
    x = fits.PrimaryHDU(frame.flux, header=hdr)
    x.header['EXTNAME'] = 'FLUX'
    hdus.append(x)

    hdus.append( fits.ImageHDU(frame.ivar, name='IVAR') )
    hdus.append( fits.ImageHDU(frame.mask, name='MASK') )
    hdus.append( fits.ImageHDU(frame.wave, name='WAVELENGTH') )
    hdus.append( fits.ImageHDU(frame.resolution_data, name='RESOLUTION' ) )
    
    hdus.writeto(outfile, clobber=True)

    return outfile

def read_frame(filename, nspec=None):
    """Reads a frame fits file and returns its data.

    Args:
        filename: path to a file, or (night, expid, camera) tuple where
            night = string YEARMMDD
            expid = integer exposure ID
            camera = b0, r1, .. z9

    Returns:
        desispec.Frame object with attributes wave, flux, ivar, etc.
    """
    #- check if filename is (night, expid, camera) tuple instead
    if not isinstance(filename, (str, unicode)):
        night, expid, camera = filename
        filename = findfile('frame', night, expid, camera)

    if not os.path.isfile(filename) :
        raise IOError("cannot open"+filename)

    fx = fits.open(filename, uint=True)
    hdr = fx[0].header
    flux = native_endian(fx['FLUX'].data)
    ivar = native_endian(fx['IVAR'].data)
    wave = native_endian(fx['WAVELENGTH'].data)
    if 'MASK' in fx:
        mask = native_endian(fx['MASK'].data)
    else:
        mask = None   #- let the Frame object create the default mask
        
    resolution_data = native_endian(fx['RESOLUTION'].data)
    fx.close()

    if nspec is not None:
        flux = flux[0:nspec]
        ivar = ivar[0:nspec]
        resolution_data = resolution_data[0:nspec]

    # return flux,ivar,wave,resolution_data, hdr
    return Frame(wave, flux, ivar, mask, resolution_data, meta=hdr)
