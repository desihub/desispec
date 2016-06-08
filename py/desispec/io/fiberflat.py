"""
desispec.io.fiberflat
=====================

IO routines for fiberflat.
"""
import os
from astropy.io import fits

from desispec.fiberflat import FiberFlat
from desispec.io import findfile
from desispec.io.util import fitsheader, native_endian, makepath

def write_fiberflat(outfile,fiberflat,header=None):
    """Write fiberflat object to outfile

    Args:
        outfile: filepath string or (night, expid, camera) tuple
        fiberflat: FiberFlat object
        header: (optional) dict or fits.Header object to use as HDU 0 header

    Returns:
        filepath of file that was written
    """
    outfile = makepath(outfile, 'fiberflat')

    if header is None:
        hdr = fitsheader(fiberflat.header)
    else:
        hdr = fitsheader(header)

    ff = fiberflat   #- shorthand
    
    hdus = fits.HDUList()
    hdus.append(fits.PrimaryHDU(ff.fiberflat.astype('f4'), header=hdr))
    hdus.append(fits.ImageHDU(ff.ivar.astype('f4'),     name='IVAR'))
    hdus.append(fits.CompImageHDU(ff.mask,              name='MASK'))
    hdus.append(fits.ImageHDU(ff.meanspec.astype('f4'), name='MEANSPEC'))
    hdus.append(fits.ImageHDU(ff.wave.astype('f4'),     name='WAVELENGTH'))
    
    hdus.writeto(outfile+'.tmp', clobber=True, checksum=True)
    os.rename(outfile+'.tmp', outfile)
    return outfile


def read_fiberflat(filename):
    """Read fiberflat from filename

    Args:
        filename (str): Name of fiberflat file, or (night, expid, camera) tuple

    Returns:
        FiberFlat object with attributes
            fiberflat, ivar, mask, meanspec, wave, header

    Notes:
        fiberflat, ivar, mask are 2D [nspec, nwave]
        meanspec and wave are 1D [nwave]
    """
    #- check if outfile is (night, expid, camera) tuple instead
    if isinstance(filename, (tuple, list)) and len(filename) == 3:
        night, expid, camera = filename
        filename = findfile('fiberflat', night, expid, camera)

    header    = fits.getheader(filename, 0)
    fiberflat = native_endian(fits.getdata(filename, 0))
    ivar      = native_endian(fits.getdata(filename, "IVAR").astype('f8'))
    mask      = native_endian(fits.getdata(filename, "MASK", uint=True))
    meanspec  = native_endian(fits.getdata(filename, "MEANSPEC").astype('f8'))
    wave      = native_endian(fits.getdata(filename, "WAVELENGTH").astype('f8'))

    return FiberFlat(wave, fiberflat, ivar, mask, meanspec, header=header)
