"""
desispec.io.frame
=================

IO routines for frame.
"""
import os.path

import numpy as np
import scipy,scipy.sparse
from astropy.io import fits

from desispec.io import findfile
from desispec.io.util import fitsheader, native_endian, makepath
from desispec.log import get_logger

log = get_logger()

def write_frame(outfile, flux, ivar, wave, resolution_data, header=None):
    """Write a frame fits file and returns path to file written.

    Args:
        outfile: full path to output file, or tuple (night, expid, channel)
        flux[nspec, nwave] : 2D object flux array
        ivar[nspec, nwave] : 2D inverse variance of flux
        wave[nwave] : 1D wavelength array in Angstroms
        resolution_data[nspec, ndiag, nwave] : optional 3D resolution matrix data
    """
    outfile = makepath(outfile, 'frame')

    hdr = fitsheader(header)

    if 'SPECMIN' not in hdr:
        hdr['SPECMIN'] = 0
    if 'SPECMAX' not in hdr:
        hdr['SPECMAX'] = hdr['SPECMIN'] + flux.shape[0]

    hdr['EXTNAME'] = ('FLUX', 'no dimension')
    fits.writeto(outfile,flux,header=hdr, clobber=True)

    hdr['EXTNAME'] = ('IVAR', 'no dimension')
    hdu = fits.ImageHDU(ivar, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)

    hdr['EXTNAME'] = ('WAVELENGTH', '[Angstroms]')
    hdu = fits.ImageHDU(wave, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)

    hdr['EXTNAME'] = ('RESOLUTION', 'no dimension')
    hdu = fits.ImageHDU(resolution_data, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)

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

    return flux,ivar,wave,resolution_data, hdr

def resolution_data_to_sparse_matrix(resolution_data,fiber=None):
    """Convert the resolution data for a given fiber into a sparse matrix.

    Use function M.todense() or M.toarray() to convert output sparse matrix M
    to a dense matrix or numpy array.
    """

    log.warning('Function desispec.io.frame.resolution_data_to_sparse_matrix is deprecated. ' +
        'Use desispec.resolution instead.')

    if len(resolution_data.shape)==3 :
        nfibers=resolution_data.shape[0]
        d=resolution_data.shape[1]/2
        nwave=resolution_data.shape[2]
        offsets = range(d,-d-1,-1)
        return scipy.sparse.dia_matrix((resolution_data[fiber],offsets),(nwave,nwave))
    elif len(resolution_data.shape)==2 :
        if fiber is not None:
            log.error("error in resolution_data_to_sparse_matrix, shape={0} and requested fiber={1}".format(str(resolution_data.shape),str(fiber)))
            sys.exit(12)
        d=resolution_data.shape[0]/2
        nwave=resolution_data.shape[1]
        offsets = np.arange(d,-d-1,-1)
        return scipy.sparse.dia_matrix((resolution_data,offsets),(nwave,nwave))
    else :
        log.error("error in resolution_data_to_sparse_matrix, shape={0}".format(str(resolution_data.shape)))
        sys.exit(12)
