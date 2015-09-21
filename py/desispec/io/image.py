"""
desispec.io.image
=================

I/O routines for Image objects
"""

import numpy as np

from desispec.image import Image
from desispec.io.util import fitsheader, native_endian, makepath
from astropy.io import fits

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
        
    hx = fits.HDUList()
    hdu = fits.ImageHDU(image.pix.astype(np.float32), name='IMAGE', header=hdr)
    if 'CAMERA' not in hdu.header:
        hdu.header.append( ('CAMERA', image.camera, 'Spectograph Camera') )

    if 'RDNOISE' not in hdu.header:
        hdu.header.append( ('RDNOISE', image.readnoise, 'Read noise [RMS electrons/pixel]'))

    hx.append(hdu)
    
    hx.append(fits.ImageHDU(image.ivar.astype(np.float32), name='IVAR'))
    hx.append(fits.CompImageHDU(image.mask.astype(np.int16), name='MASK'))
    hx.writeto(outfile, clobber=True)
    
    return outfile
    
def read_image(filename):
    """
    Returns desispec.image.Image object from input file
    """
    fx = fits.open(filename, uint=True)
    image = native_endian(fx['IMAGE'].data).astype(np.float64)
    ivar = native_endian(fx['IVAR'].data).astype(np.float64)
    mask = native_endian(fx['MASK'].data).astype(np.uint16)
    readnoise = fx['IMAGE'].header['RDNOISE']
    camera = fx['IMAGE'].header['CAMERA']
    meta = fx['IMAGE'].header
    
    return Image(image, ivar, mask=mask, readnoise=readnoise,
                 camera=camera, meta=meta)
