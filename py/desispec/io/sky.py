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
            flux : 1D unconvolved sky flux
            ivar : inverse variance of skyflux
            mask : mask for skyflux
            cflux : 1D skyflux convolved with the mean resolution across all fibers
            civar : inverse variance of cskyflux
        header : optional fits header data (fits.Header, dict, or list)
    """
    outfile = makepath(outfile, 'sky')

    #- Convert header to fits.Header if needed
    hdr = fitsheader(header)

    hdr['EXTNAME'] = ('SKY', 'no dimension')
    fits.writeto(outfile, skymodel.flux,header=hdr, clobber=True)

    hdr['EXTNAME'] = ('IVAR', 'no dimension')
    hdu = fits.ImageHDU(skymodel.ivar, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)

    hdr['EXTNAME'] = ('MASK', 'no dimension')
    hdu = fits.ImageHDU(skymodel.mask, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)

    hdr['EXTNAME'] = ('CSKY', 'convolved sky at average resolution')
    hdu = fits.ImageHDU(skymodel.cflux, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)

    hdr['EXTNAME'] = ('CIVAR', 'convolved sky inverse variance')
    hdu = fits.ImageHDU(skymodel.civar, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)

    hdr['EXTNAME'] = ('WAVELENGTH', '[Angstroms]')
    hdu = fits.ImageHDU(skymodel.wave, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)

    return outfile

def read_sky(filename) :
    """Read sky model and return SkyModel object with attributes
    flux, ivar, mask, cflux, civar, wave, header.

    These are 1D unconvolved arrays that need to be convolved with the
    per fiber resolution matrix to get the sky model for each fiber.

    cflux & civar are the convolved quanities at mean resolution.
    """
    #- check if filename is (night, expid, camera) tuple instead
    if not isinstance(filename, (str, unicode)):
        night, expid, camera = filename
        filename = findfile('sky', night, expid, camera)

    hdr = fits.getheader(filename, 0)
    wave = native_endian(fits.getdata(filename, "WAVELENGTH"))
    skyflux = native_endian(fits.getdata(filename, "SKY"))
    ivar = native_endian(fits.getdata(filename, "IVAR"))
    mask = native_endian(fits.getdata(filename, "MASK"))
    cskyflux = native_endian(fits.getdata(filename, "CSKY"))
    civar = native_endian(fits.getdata(filename, "CIVAR"))

    skymodel = SkyModel(wave,skyflux,ivar,mask,cskyflux,civar)
    skymodel.header = hdr

    return skymodel