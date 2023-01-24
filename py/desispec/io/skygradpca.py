"""
desispec.io.skygradpca
======================

IO routines for sky gradient pca.
"""

from __future__ import absolute_import, division
import os
import time
from astropy.io import fits
from desiutil.log import get_logger

from . import iotime
from .util import get_tempfilename

def write_skygradpca(outfile, skygradpca):
    """Write sky model.

    Args:
        outfile : filename or (night, expid, camera) tuple
        skygradpca : SkyGradPCA object
    """
    from .util import fitsheader, makepath

    log = get_logger()
    outfile = makepath(outfile, 'skygradpca')

    # Convert header to fits.Header if needed
    hdr = fitsheader(skygradpca.header)

    hx = fits.HDUList()
    hdr['EXTNAME'] = 'FLUX'
    hx.append(fits.PrimaryHDU(skygradpca.flux.astype('f4'), header=hdr))
    hx.append(fits.ImageHDU(skygradpca.wave.astype('f8'), name='WAVELENGTH'))

    t0 = time.time()
    tmpfile = get_tempfilename(outfile)
    hx.writeto(tmpfile, overwrite=True, checksum=True)
    os.rename(tmpfile, outfile)
    duration = time.time() - t0
    log.info(iotime.format('write', outfile, duration))
    return outfile


def read_skygradpca(filename):
    """Read sky grad pca file and return SkyGradPCA object.
    """
    from .util import native_endian
    from ..skygradpca import SkyGradPCA

    log = get_logger()
    t0 = time.time()
    fx = fits.open(filename, memmap=False, uint=True)

    hdr = fx[0].header
    wave = native_endian(fx["WAVELENGTH"].data.astype('f8'))
    flux = native_endian(fx["FLUX"].data.astype('f4'))
    fx.close()

    duration = time.time() - t0
    log.info(iotime.format('read', filename, duration))

    return SkyGradPCA(wave=wave, flux=flux, header=hdr)
