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

def write_sky(outfile, skymodel, header=None):
    """Write sky model.

    Args:
        outfile : filename or (night, expid, camera) tuple
        skymodel : SkyModel object, with the following attributes
            wave : 1D wavelength in vacuum Angstroms
            flux : 2D[nspec, nwave] sky flux
            ivar : 2D inverse variance of sky flux
            mask : 2D mask for sky flux
            stat_ivar : 2D inverse variance of sky flux (statistical only)
        header : optional fits header data (fits.Header, dict, or list)
    """
    from desiutil.depend import add_dependencies
    from .util import fitsheader, makepath

    log = get_logger()
    outfile = makepath(outfile, 'sky')

    #- Convert header to fits.Header if needed
    if header is not None:
        hdr = fitsheader(header)
    else:
        hdr = fitsheader(skymodel.header)

    add_dependencies(hdr)

    hx = fits.HDUList()

    hdr['EXTNAME'] = ('SKY', 'no dimension')
    hx.append( fits.PrimaryHDU(skymodel.flux.astype('f4'), header=hdr) )
    hx.append( fits.ImageHDU(skymodel.ivar.astype('f4'), name='IVAR') )
    # hx.append( fits.CompImageHDU(skymodel.mask, name='MASK') )
    hx.append( fits.ImageHDU(skymodel.mask, name='MASK') )
    hx.append( fits.ImageHDU(skymodel.wave.astype('f4'), name='WAVELENGTH') )
    if skymodel.stat_ivar is not None :
       hx.append( fits.ImageHDU(skymodel.stat_ivar.astype('f4'), name='STATIVAR') )
    if skymodel.throughput_corrections is not None:
        hx.append( fits.ImageHDU(skymodel.throughput_corrections.astype('f4'), name='THRPUTCORR') )

    hx[-1].header['BUNIT'] = 'Angstrom'

    t0 = time.time()
    hx.writeto(outfile+'.tmp', overwrite=True, checksum=True)
    os.rename(outfile+'.tmp', outfile)
    iotime = time.time() - t0
    log.info(f'iotime {iotime:.3f} sec to write {outfile}')

    return outfile

def read_sky(filename) :
    """Read sky model and return SkyModel object with attributes
    wave, flux, ivar, mask, header.

    skymodel.wave is 1D common wavelength grid, the others are 2D[nspec, nwave]
    """
    from .meta import findfile
    from .util import native_endian
    from ..sky import SkyModel
    log = get_logger()
    #- check if filename is (night, expid, camera) tuple instead
    if not isinstance(filename, str):
        night, expid, camera = filename
        filename = findfile('sky', night, expid, camera)

    t0 = time.time()
    fx = fits.open(filename, memmap=False, uint=True)

    hdr = fx[0].header
    wave = native_endian(fx["WAVELENGTH"].data.astype('f8'))
    skyflux = native_endian(fx["SKY"].data.astype('f8'))
    ivar = native_endian(fx["IVAR"].data.astype('f8'))
    mask = native_endian(fx["MASK"].data)
    if "STATIVAR" in fx :
        stat_ivar = native_endian(fx["STATIVAR"].data.astype('f8'))
    else :
        stat_ivar = None
    if "THRPUTCORR" in fx :
        throughput_corrections = native_endian(fx["THRPUTCORR"].data.astype('f8'))
    else :
        throughput_corrections = None
    fx.close()
    iotime = time.time() - t0
    log.info(f'iotime {iotime:.3f} sec to read {filename}')

    skymodel = SkyModel(wave, skyflux, ivar, mask, header=hdr,stat_ivar=stat_ivar,\
                        throughput_corrections=throughput_corrections)

    return skymodel
