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
import numpy as np

from . import iotime
from .util import get_tempfilename

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
    hx.append( fits.PrimaryHDU(skycorr.wave.astype('f8'), header=hdr) )
    hx.append( fits.ImageHDU(skycorr.dwave.astype('f4'), name='DWAVE') )
    hx.append( fits.ImageHDU(skycorr.dlsf.astype('f4'), name='DLSF') )
    hx['DWAVE'].header['BUNIT'] = 'Angstrom'
    hx['DLSF'].header['BUNIT'] = 'Angstrom'
    t0 = time.time()
    tmpfile = get_tempfilename(outfile)
    hx.writeto(tmpfile, overwrite=True, checksum=True)
    os.rename(tmpfile, outfile)
    duration = time.time() - t0
    log.info(iotime.format('write', outfile, duration))
    return outfile

def read_skycorr(filename) :
    """Read sky correction and return SkyCorr object with attributes
    wave, flux, ivar, mask, header.

    skymodel.wave is 1D common wavelength grid, the others are 2D[nspec, nwave]
    """
    from .meta import findfile
    from .util import native_endian, checkgzip
    from ..skycorr import SkyCorr
    log = get_logger()
    #- check if filename is (night, expid, camera) tuple instead
    if not isinstance(filename, str):
        night, expid, camera = filename
        filename = findfile('skycorr', night, expid, camera)

    t0 = time.time()
    filename = checkgzip(filename)
    fx = fits.open(filename, memmap=False, uint=True)

    hdr = fx[0].header
    wave  = native_endian(fx["WAVELENGTH"].data.astype('f8'))
    dwave = native_endian(fx["DWAVE"].data.astype('f8'))
    dlsf  = native_endian(fx["DLSF"].data.astype('f8'))
    fx.close()
    duration = time.time() - t0
    log.info(iotime.format('read', filename, duration))

    return SkyCorr(wave, dwave, dlsf, header=hdr)

def write_skycorr_pca(outfile, skycorrpca):
    """Write sky model.

    Args:
        outfile : filename or (night, expid, camera) tuple
        skycorrpca : SkyCorrPCA object
    """
    from .util import fitsheader, makepath

    log = get_logger()
    outfile = makepath(outfile, 'skycorr')

    #- Convert header to fits.Header if needed
    hdr = fitsheader(skycorrpca.header)

    hx = fits.HDUList()
    hdr['EXTNAME'] = 'DWAVE_MEAN'
    hx.append( fits.PrimaryHDU(skycorrpca.dwave_mean.astype('f4'), header=hdr) )
    for i in range(skycorrpca.dwave_eigenvectors.shape[0]) :
        hx.append( fits.ImageHDU(skycorrpca.dwave_eigenvectors[i].astype('f4'), name='DWAVE_EIG{}'.format(i+1)))
    hx.append( fits.ImageHDU(skycorrpca.dwave_eigenvalues.astype('f8'), name='DWAVE_EIGENVALS'))

    hx.append( fits.ImageHDU(skycorrpca.dlsf_mean.astype('f4'), name='DLSF_MEAN'))
    for i in range(skycorrpca.dlsf_eigenvectors.shape[0]) :
        hx.append( fits.ImageHDU(skycorrpca.dlsf_eigenvectors[i].astype('f4'), name='DLSF_EIG{}'.format(i+1)))
    hx.append( fits.ImageHDU(skycorrpca.dlsf_eigenvalues.astype('f8'), name='DLSF_EIGENVALS'))

    hx.append( fits.ImageHDU(skycorrpca.wave.astype('f8'), name='WAVELENGTH'))

    t0 = time.time()
    tmpfile = get_tempfilename(outfile)
    hx.writeto(tmpfile, overwrite=True, checksum=True)
    os.rename(tmpfile, outfile)
    duration = time.time() - t0
    log.info(iotime.format('write', outfile, duration))
    return outfile

def read_skycorr_pca(filename) :
    """Read sky correction pca file and return SkyCorrPCA object.
    """
    from .meta import findfile
    from .util import native_endian, checkgzip
    from ..skycorr import SkyCorrPCA

    log = get_logger()
    t0 = time.time()
    filename = checkgzip(filename)
    fx = fits.open(filename, memmap=False, uint=True)

    hdr = fx[0].header
    wave  = native_endian(fx["WAVELENGTH"].data.astype('f8'))
    dwave_mean = native_endian(fx["DWAVE_MEAN"].data.astype('f8'))
    dwave_eigenvectors = []
    for i in range(1,12):
        key="DWAVE_EIG{}".format(i)
        if key in fx :
            dwave_eigenvectors.append(fx[key].data.astype('f8'))
        else:
            break
    dwave_eigenvectors = np.array(dwave_eigenvectors)
    dwave_eigenvalues = native_endian(fx["DWAVE_EIGENVALS"].data.astype('f8'))

    dlsf_mean = native_endian(fx["DLSF_MEAN"].data.astype('f8'))
    dlsf_eigenvectors = []
    for i in range(1,12):
        key="DLSF_EIG{}".format(i)
        if key in fx :
            dlsf_eigenvectors.append(fx[key].data.astype('f8'))
        else:
            break
    dlsf_eigenvectors = np.array(dlsf_eigenvectors)

    dlsf_eigenvalues = native_endian(fx["DLSF_EIGENVALS"].data.astype('f8'))

    fx.close()
    duration = time.time() - t0
    log.info(iotime.format('read', filename, duration))

    return SkyCorrPCA(dwave_mean=dwave_mean,
                      dwave_eigenvectors=dwave_eigenvectors,
                      dwave_eigenvalues=dwave_eigenvalues,
                      dlsf_mean=dlsf_mean,
                      dlsf_eigenvectors=dlsf_eigenvectors,
                      dlsf_eigenvalues=dlsf_eigenvalues,
                      wave=wave,
                      header=hdr)
