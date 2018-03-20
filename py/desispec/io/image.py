"""
desispec.io.image
=================

I/O routines for Image objects
"""

import os
import numpy as np

from desispec.image import Image
from desispec.io.util import fitsheader, native_endian, makepath
from astropy.io import fits
from desiutil.depend import add_dependencies

def write_image(outfile, image, meta=None):
    """Writes image object to outfile
    
    Args:
        outfile : output file string
        image : desispec.image.Image object
            (or any object with 2D array attributes image, ivar, mask)
    
    Optional:
        meta : dict-like object with metadata key/values (e.g. FITS header)
    """

    if meta is not None:
        hdr = fitsheader(meta)
    else:
        hdr = fitsheader(image.meta)

    add_dependencies(hdr)

    outdir = os.path.dirname(os.path.abspath(outfile))
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    hx = fits.HDUList()
    hdu = fits.ImageHDU(image.pix.astype(np.float32), name='IMAGE', header=hdr)
    if 'CAMERA' not in hdu.header:
        hdu.header.append( ('CAMERA', image.camera.lower(), 'Spectrograph Camera') )

    if 'RDNOISE' not in hdu.header and np.isscalar(image.readnoise):
        hdu.header.append( ('RDNOISE', image.readnoise, 'Read noise [RMS electrons/pixel]'))

    hx.append(hdu)
    hx.append(fits.ImageHDU(image.ivar.astype(np.float32), name='IVAR'))
    hx.append(fits.CompImageHDU(image.mask.astype(np.int16), name='MASK'))
    if not np.isscalar(image.readnoise):
        hx.append(fits.ImageHDU(image.readnoise.astype(np.float32), name='READNOISE'))

    hx.writeto(outfile+'.tmp', clobber=True, checksum=True)
    os.rename(outfile+'.tmp', outfile)

    return outfile

def read_image(filename):
    """
    Returns desispec.image.Image object from input file
    """
    fx = fits.open(filename, uint=True, memmap=False)
    image = native_endian(fx['IMAGE'].data).astype(np.float64)
    ivar = native_endian(fx['IVAR'].data).astype(np.float64)
    mask = native_endian(fx['MASK'].data).astype(np.uint16)
    camera = fx['IMAGE'].header['CAMERA'].lower()
    meta = fx['IMAGE'].header

    if 'READNOISE' in fx:
        readnoise = native_endian(fx['READNOISE'].data).astype(np.float64)
    else:
        readnoise = fx['IMAGE'].header['RDNOISE']

    fx.close()
    return Image(image, ivar, mask=mask, readnoise=readnoise,
                 camera=camera, meta=meta)
