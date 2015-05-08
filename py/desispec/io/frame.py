"""
desispec.io.frame
=================

IO routines for frame.
"""
import os.path

import numpy as np
import scipy,scipy.sparse
from astropy.io import fits

from desispec.spectra import Spectra
from desispec.io import findfile
from desispec.io.util import fitsheader, native_endian, makepath
from desispec.log import get_logger

log = get_logger()

def write_frame(outfile, spectra, header=None):
    """Write a frame fits file and returns path to file written.

    Args:
        outfile: full path to output file, or tuple (night, expid, channel)
        spectra: output desispec.spectra.Spectra object with wave, flux, ivar...
        header: optional astropy.io.fits.Header or dict to override spectra.header
        
    Note:
        spectra = Spectra(wave, flux, ivar, resolution_data)
    """
    outfile = makepath(outfile, 'frame')

    if header is not None:
        hdr = fitsheader(header)
    else:
        hdr = fitsheader(spectra.header)

    hdus = fits.HDUList()
    x = fits.PrimaryHDU(spectra.flux, header=hdr)
    x.header['EXTNAME'] = 'FLUX'
    hdus.append(x)

    hdus.append( fits.ImageHDU(spectra.ivar, name='IVAR') )
    hdus.append( fits.ImageHDU(spectra.wave, name='WAVELENGTH') )
    hdus.append( fits.ImageHDU(spectra.resolution_data, name='RESOLUTION' ) )
    
    hdus.writeto(outfile, clobber=True)

    return outfile

def read_frame(filename, nspec=None):
    """Reads a frame fits file and returns its data.

    Args:
        filename: path to a file, or (night, expid, camera) tuple where
            night = string YEARMMDD
            expid = integer exposure ID
            camera = b0, r1, .. z9

    Returns
        read_frame (tuple):
            phot[nspec, nwave] : uncalibrated photons per bin
            ivar[nspec, nwave] : inverse variance of phot
            wave[nwave] : vacuum wavelengths [Angstrom]
            resolution[nspec, ndiag, nwave] : TODO DOCUMENT THIS FORMAT
            header : fits.Header from HDU 0
    """
    #- check if filename is (night, expid, camera) tuple instead
    if not isinstance(filename, (str, unicode)):
        night, expid, camera = filename
        filename = findfile('frame', night, expid, camera)

    if not os.path.isfile(filename) :
        raise IOError("cannot open"+filename)

    hdr = fits.getheader(filename)
    flux = native_endian(fits.getdata(filename, 0))
    ivar = native_endian(fits.getdata(filename, "IVAR"))
    wave = native_endian(fits.getdata(filename, "WAVELENGTH"))
    resolution_data = native_endian(fits.getdata(filename, "RESOLUTION"))

    if nspec is not None:
        flux = flux[0:nspec]
        ivar = ivar[0:nspec]
        resolution_data = resolution_data[0:nspec]

    # return flux,ivar,wave,resolution_data, hdr
    return Spectra(wave, flux, ivar, resolution_data, hdr)
