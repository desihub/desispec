"""
desispec.io.sky
===============

IO routines for sky.
"""
import os
from astropy.io import fits

from desiutil.depend import add_dependencies

from desispec.sky import SkyModel
from desispec.io import findfile
from desispec.io.util import fitsheader, native_endian, makepath

def write_sky(outfile, skymodel, header=None):
    """Write sky model.

    Args:
        outfile : filename or (night, expid, camera) tuple
        skymodel : SkyModel object, with the following attributes
            wave : 1D wavelength in vacuum Angstroms
            flux : 2D[nspec, nwave] sky flux
            ivar : 2D inverse variance of sky flux
            mask : 2D mask for sky flux
        header : optional fits header data (fits.Header, dict, or list)
    """
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
    hx.append( fits.CompImageHDU(skymodel.mask, name='MASK') )
    hx.append( fits.ImageHDU(skymodel.wave.astype('f4'), name='WAVELENGTH') )

    hx.writeto(outfile+'.tmp', clobber=True, checksum=True)
    os.rename(outfile+'.tmp', outfile)

    return outfile

def read_sky(filename) :
    """Read sky model and return SkyModel object with attributes
    wave, flux, ivar, mask, header.
    
    skymodel.wave is 1D common wavelength grid, the others are 2D[nspec, nwave]
    """
    #- check if filename is (night, expid, camera) tuple instead
    if not isinstance(filename, str):
        night, expid, camera = filename
        filename = findfile('sky', night, expid, camera)

    fx = fits.open(filename, memmap=False, uint=True)

    hdr = fx[0].header
    wave = native_endian(fx["WAVELENGTH"].data.astype('f8'))
    skyflux = native_endian(fx["SKY"].data.astype('f8'))
    ivar = native_endian(fx["IVAR"].data.astype('f8'))
    mask = native_endian(fx["MASK"].data)
    fx.close()

    skymodel = SkyModel(wave, skyflux, ivar, mask, header=hdr)

    return skymodel