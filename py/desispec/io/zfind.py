"""
desispec.io.zfind
=================

IO routines for zfind.
"""

import numpy as np
from astropy.io import fits
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
        ('TYPE',      zfind.type.dtype),
        ('SUBTYPE',   zfind.subtype.dtype),
    ]

    data = np.empty(zfind.nspec, dtype=dtype)
    data['BRICKNAME'] = brickname
    data['TARGETID']  = targetids
    data['Z']         = zfind.z
    data['ZERR']      = zfind.zerr
    data['ZWARN']     = zfind.zwarn
    data['TYPE']      = zfind.type
    data['SUBTYPE']   = zfind.subtype

    hdus = fits.HDUList()
    hdus.append(fits.BinTableHDU(data, name='ZBEST', uint=True))

    if zspec:
        hdus.append(fits.ImageHDU(zfind.wave, name='WAVELENGTH'))
        hdus.append(fits.ImageHDU(zfind.flux, name='FLUX'))
        hdus.append(fits.ImageHDU(zfind.ivar, name='IVAR'))
        hdus.append(fits.ImageHDU(zfind.model, name='MODEL'))

    hdus.writeto(filename, clobber=True)


def read_zbest(filename):
    """Returns a desispec.zfind.ZfindBase object with contents from filename.
    """
    fx = fits.open(filename)
    zbest = fx['ZBEST'].data
    if 'WAVELENGTH' in fx:
        wave = fx['WAVELENGTH'].data
        flux = fx['FLUX'].data
        ivar = fx['IVAR'].data
        model = fx['MODEL'].data

        zf = ZfindBase(wave, flux, ivar, results=zbest)
        zf.model = model
    else:
        zf = ZfindBase(None, None, None, results=zbest)

    fx.close()
    return zf

#- TODO: This should be moved to a separate test file.
def _test_zbest_io():
    import os
    log=get_logger()
    nspec, nflux = 10, 20
    wave = np.arange(nflux)
    flux = np.random.uniform(size=(nspec, nflux))
    ivar = np.random.uniform(size=(nspec, nflux))
    zfind1 = ZfindBase(wave, flux, ivar)

    brickname = '1234p567'
    targetids = np.random.randint(0,12345678, size=nspec)

    outfile = 'zbest_test.fits'
    write_zbest(outfile, brickname, targetids, zfind1)
    zfind2 = read_zbest(outfile)

    assert np.all(zfind2.z == zfind1.z)
    assert np.all(zfind2.zerr == zfind1.zerr)
    assert np.all(zfind2.zwarn == zfind1.zwarn)
    assert np.all(zfind2.type == zfind1.type)
    assert np.all(zfind2.subtype == zfind1.subtype)
    assert np.all(zfind2.brickname == brickname)
    assert np.all(zfind2.targetid == targetids)

    write_zbest(outfile, brickname, targetids, zfind1, zspec=True)
    zfind3 = read_zbest(outfile)
    assert np.all(zfind3.wave == zfind1.wave)
    assert np.all(zfind3.flux == zfind1.flux)
    assert np.all(zfind3.ivar == zfind1.ivar)
    assert np.all(zfind3.model == zfind1.model)

    log.info("looks OK to me")

    os.remove(outfile)
