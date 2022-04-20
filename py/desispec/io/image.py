"""
desispec.io.image
=================

I/O routines for Image objects
"""

import os
import time
import warnings
import numpy as np

from desispec.image import Image
from desispec.io.util import fitsheader, native_endian, makepath
from . import iotime
from .util import checkgzip, get_tempfilename
from astropy.io import fits
from desiutil.depend import add_dependencies
from desiutil.log import get_logger
from astropy.table import Table

def write_image(outfile, image, meta=None):
    """Writes image object to outfile

    Args:
        outfile : output file string
        image : desispec.image.Image object
            (or any object with 2D array attributes image, ivar, mask)

    Optional:
        meta : dict-like object with metadata key/values (e.g. FITS header)
    """

    log = get_logger()
    if meta is not None:
        hdr = fitsheader(meta)
    else:
        hdr = fitsheader(image.meta)

    add_dependencies(hdr)

    #- Work around fitsio>1.0 writing blank keywords, e.g. on 20191212
    for key in hdr.keys():
        if type(hdr[key]) == fits.card.Undefined:
            log.warning('Setting blank keyword {} to None'.format(key))
            hdr[key] = None

    outdir = os.path.dirname(os.path.abspath(outfile))
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

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

    if hasattr(image, 'fibermap'):
        if isinstance(image.fibermap, Table):
            with warnings.catch_warnings():
                #- nanomaggies aren't an official IAU unit but don't complain
                warnings.filterwarnings('ignore', ".*nanomaggies.*")
                fmhdu = fits.convenience.table_to_hdu(image.fibermap)
            fmhdu.name = 'FIBERMAP'
        else:
            fmhdu = fits.BinTableHDU(image.fibermap, name='FIBERMAP')

        hx.append(fmhdu)

    t0 = time.time()
    tmpfile = get_tempfilename(outfile)
    hx.writeto(tmpfile, overwrite=True, checksum=True)
    os.rename(tmpfile, outfile)
    duration = time.time() - t0
    log.info(iotime.format('write', outfile, duration))

    return outfile

def read_image(filename):
    """
    Returns desispec.image.Image object from input file
    """
    log = get_logger()
    filename = checkgzip(filename)
    t0 = time.time()
    with fits.open(filename, uint=True, memmap=False) as fx:
        image = native_endian(fx['IMAGE'].data).astype(np.float64)
        ivar = native_endian(fx['IVAR'].data).astype(np.float64)
        mask = native_endian(fx['MASK'].data).astype(np.uint16)
        camera = fx['IMAGE'].header['CAMERA'].lower()
        meta = fx['IMAGE'].header

        if 'READNOISE' in fx:
            readnoise = native_endian(fx['READNOISE'].data).astype(np.float64)
        else:
            readnoise = fx['IMAGE'].header['RDNOISE']

    duration = time.time() - t0
    log.info(iotime.format('read', filename, duration))

    return Image(image, ivar, mask=mask, readnoise=readnoise,
                 camera=camera, meta=meta)
