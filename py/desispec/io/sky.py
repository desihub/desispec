"""
desispec.io.sky
===============

IO routines for sky.
"""
import os
from astropy.io import fits

from desispec.io import findfile
from desispec.io.util import fitsheader, native_endian, makepath

def write_sky(outfile,skyflux,skyivar,skymask,cskyflux,cskyivar,wave, header=None):
    """Write sky model.

    Args:
        outfile : filename or (night, expid, camera) tuple
        skyflux : 1D unconvolved sky flux
        skyivar : inverse variance of skyflux
        skymask : mask for skyflux
        cskyflux : 1D skyflux convolved with the mean resolution across all fibers
        cskyivar : inverse variance of cskyflux
        wave : 1D wavelength in vacuum Angstroms
        header : optional fits header data (fits.Header, dict, or list)
    """
    outfile = makepath(outfile, 'sky')

    #- Convert header to fits.Header if needed
    hdr = fitsheader(header)

    hdr['EXTNAME'] = ('SKY', 'no dimension')
    fits.writeto(outfile,skyflux,header=hdr, clobber=True)

    hdr['EXTNAME'] = ('IVAR', 'no dimension')
    hdu = fits.ImageHDU(skyivar, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)

    hdr['EXTNAME'] = ('MASK', 'no dimension')
    hdu = fits.ImageHDU(skymask, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)

    hdr['EXTNAME'] = ('CSKY', 'convolved sky at average resolution')
    hdu = fits.ImageHDU(cskyflux, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)

    hdr['EXTNAME'] = ('CIVAR', 'convolved sky inverse variance')
    hdu = fits.ImageHDU(cskyivar, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)

    hdr['EXTNAME'] = ('WAVELENGTH', '[Angstroms]')
    hdu = fits.ImageHDU(wave, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)

    return outfile

def read_sky(filename) :
    """Read sky model and return tuple of (skyflux, ivar, mask, cskyflux, civar, wave, header).

    These are 1D unconvolved arrays that need to be convolved with the
    per fiber resolution matrix to get the sky model for each fiber.

    cskyflux & civar are the convolved quanities at mean resolution.
    """
    #- check if filename is (night, expid, camera) tuple instead
    if not isinstance(filename, (str, unicode)):
        night, expid, camera = filename
        filename = findfile('sky', night, expid, camera)

    hdr = fits.getheader(filename, 0)
    skyflux = native_endian(fits.getdata(filename, "SKY"))
    ivar = native_endian(fits.getdata(filename, "IVAR"))
    mask = native_endian(fits.getdata(filename, "MASK"))
    cskyflux = native_endian(fits.getdata(filename, "CSKY"))
    civar = native_endian(fits.getdata(filename, "CIVAR"))
    wave = native_endian(fits.getdata(filename, "WAVELENGTH"))

    return skyflux,ivar,mask,cskyflux,civar,wave,hdr
