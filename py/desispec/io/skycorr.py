"""
desispec.io.sky
===============

IO routines for sky.
"""
from __future__ import absolute_import, division
import os
import time
from astropy.io import fits
from desiutil.log import get_logger

from . import iotime

def write_skycorr(outfile, skycorr):
    """Write sky model.

    Args:
        outfile : filename or (night, expid, camera) tuple
        skycorr : SkyCorr object
    """
    from .util import fitsheader, makepath

    log = get_logger()
    outfile = makepath(outfile, 'skycorr')

    #- Convert header to fits.Header if needed
    hdr = fitsheader(skycorr.header)

    hx = fits.HDUList()
    hdr['EXTNAME'] = 'WAVELENGTH'
    hx.append( fits.PrimaryHDU(skycorr.wave.astype('f4'), header=hdr) )
    hx.append( fits.ImageHDU(skycorr.dwave.astype('f4'), name='DWAVE') )
    hx.append( fits.ImageHDU(skycorr.dlsf.astype('f4'), name='DLSF') )
    hx['DWAVE'].header['BUNIT'] = 'Angstrom'
    hx['DLSF'].header['BUNIT'] = 'Angstrom'
    t0 = time.time()
    hx.writeto(outfile+'.tmp', overwrite=True, checksum=True)
    os.rename(outfile+'.tmp', outfile)
    duration = time.time() - t0
    log.info(iotime.format('write', outfile, duration))
    return outfile

def read_skycorr(filename) :
    """Read sky correction and return SkyCorr object with attributes
    wave, flux, ivar, mask, header.

    skymodel.wave is 1D common wavelength grid, the others are 2D[nspec, nwave]
    """
    from .meta import findfile
    from .util import native_endian
    from ..skycorr import SkyCorr
    log = get_logger()
    #- check if filename is (night, expid, camera) tuple instead
    if not isinstance(filename, str):
        night, expid, camera = filename
        filename = findfile('skycorr', night, expid, camera)

    t0 = time.time()
    fx = fits.open(filename, memmap=False, uint=True)

    hdr = fx[0].header
    wave  = native_endian(fx["WAVELENGTH"].data.astype('f8'))
    dwave = native_endian(fx["DWAVE"].data.astype('f8'))
    dlsf  = native_endian(fx["DLSF"].data.astype('f8'))
    fx.close()
    duration = time.time() - t0
    log.info(iotime.format('read', filename, duration))

    return SkyCorr(wave, dwave, dlsf, header=hdr)
