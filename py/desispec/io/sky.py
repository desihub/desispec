"""
desispec.io.sky
===============

IO routines for sky.
"""
import os
from astropy.io import fits

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

    hdr['EXTNAME'] = ('SKY', 'no dimension')
    fits.writeto(outfile, skymodel.flux,header=hdr, clobber=True)

    hdr['EXTNAME'] = ('IVAR', 'no dimension')
    hdu = fits.ImageHDU(skymodel.ivar, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)

    hdr['EXTNAME'] = ('MASK', 'no dimension')
    hdu = fits.ImageHDU(skymodel.mask, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)

    hdr['EXTNAME'] = ('WAVELENGTH', '[Angstroms]')
    hdu = fits.ImageHDU(skymodel.wave, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)

    return outfile

def read_sky(filename) :
    """Read sky model and return SkyModel object with attributes
    wave, flux, ivar, mask, header.
    
    skymodel.wave is 1D common wavelength grid, the others are 2D[nspec, nwave]
    """
    #- check if filename is (night, expid, camera) tuple instead
    if not isinstance(filename, (str, unicode)):
        night, expid, camera = filename
        filename = findfile('sky', night, expid, camera)

    hdr = fits.getheader(filename, 0)
    wave = native_endian(fits.getdata(filename, "WAVELENGTH"))
    skyflux = native_endian(fits.getdata(filename, "SKY"))
    ivar = native_endian(fits.getdata(filename, "IVAR"))
    mask = native_endian(fits.getdata(filename, "MASK", uint=True))

    skymodel = SkyModel(wave, skyflux, ivar, mask, header=hdr)

    return skymodel