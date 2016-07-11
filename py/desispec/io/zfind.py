"""
desispec.io.zfind
=================

IO routines for zfind.
"""

import os

import numpy as np
from astropy.io import fits
from desiutil.depend import add_dependencies
from desispec.zfind import ZfindBase
from desispec.log import get_logger

def write_zbest(filename, brickname, targetids, zfind, zspec=False):
    """Writes zfinder output to ``filename``.

    Args:
        filename : full path to output file
        brickname : brick name, e.g. '1234p5678'
        targetids[nspec] : 1D array of target IDs
        zfind : subclass of desispec.zfind.ZFindBase with member variables
            z, zerr, zwarn, type, subtype
        zspec : if True, also write the following member variables from zfind
            wave[nwave], flux[nspec, nwave], ivar[nspec, nwave],
            model[nspec, nwave]

    The first set of variables are each 1D arrays and are written into a
    binary table in an HDU with EXTNAME=ZBEST.  The zspec=True outputs are
    written to 3 additional image HDUs with names WAVELENGTH, FLUX, IVAR,
    and MODEL.
    """
    dtype = [
        ('BRICKNAME', 'S8'),
        ('TARGETID',  np.int64),
        ('Z',         zfind.z.dtype),
        ('ZERR',      zfind.zerr.dtype),
        ('ZWARN',     zfind.zwarn.dtype),
        ('SPECTYPE',  zfind.spectype.dtype),
        ('SUBTYPE',   zfind.subtype.dtype),
    ]

    data = np.empty(zfind.nspec, dtype=dtype)
    data['BRICKNAME'] = brickname
    data['TARGETID']  = targetids
    data['Z']         = zfind.z
    data['ZERR']      = zfind.zerr
    data['ZWARN']     = zfind.zwarn
    data['SPECTYPE']  = zfind.spectype
    data['SUBTYPE']   = zfind.subtype

    hdus = fits.HDUList()
    phdr = fits.Header()
    add_dependencies(phdr)
    hdus.append(fits.PrimaryHDU(None, header=phdr))
    hdus.append(fits.BinTableHDU(data, name='ZBEST', uint=True))

    if zspec:
        hdus.append(fits.ImageHDU(zfind.wave.astype('f4'), name='WAVELENGTH'))
        hdus.append(fits.ImageHDU(zfind.flux.astype('f4'), name='FLUX'))
        hdus.append(fits.ImageHDU(zfind.ivar.astype('f4'), name='IVAR'))
        hdus.append(fits.ImageHDU(zfind.model.astype('f4'), name='MODEL'))

    hdus.writeto(filename+'.tmp', clobber=True, checksum=True)
    os.rename(filename+'.tmp', filename)


def read_zbest(filename):
    """Returns a desispec.zfind.ZfindBase object with contents from filename.
    """
    from desispec.io.util import native_endian
    fx = fits.open(filename, memmap=False)
    zbest = fx['ZBEST'].data
    if 'WAVELENGTH' in fx:
        wave = native_endian(fx['WAVELENGTH'].data.astype('f8'))
        flux = native_endian(fx['FLUX'].data.astype('f8'))
        ivar = native_endian(fx['IVAR'].data.astype('f8'))
        model = fx['MODEL'].data

        zf = ZfindBase(wave, flux, ivar, results=zbest)
        zf.model = model
    else:
        zf = ZfindBase(None, None, None, results=zbest)

    fx.close()
    return zf
