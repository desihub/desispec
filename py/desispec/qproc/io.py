"""
desispec.qproc.io
=================

I/O routines for qproc objects
"""

import os.path

import numpy as np
import scipy, scipy.sparse
from astropy.io import fits
import warnings

from desiutil.depend import add_dependencies
from desiutil.io import encode_table

from desispec.qproc.qframe import QFrame
from desispec.io.util import fitsheader, native_endian, makepath
from desiutil.log import get_logger


def write_qframe(outfile, qframe, header=None, fibermap=None, units=None):
    """Write a frame fits file and returns path to file written.

    Args:
        outfile: full path to output file, or tuple (night, expid, channel)
        qframe:  desispec.qproc.QFrame object with wave, flux, ivar...

    Optional:
        header: astropy.io.fits.Header or dict to override frame.header
        fibermap: table to store as FIBERMAP HDU

    Returns:
        full filepath of output file that was written

    Note:
        to create a QFrame object to pass into write_qframe,
        qframe = QFrame(wave, flux, ivar)
    """
    log = get_logger()
    outfile = makepath(outfile, 'qframe')

    if header is not None:
        hdr = fitsheader(header)
    else:
        hdr = fitsheader(qframe.meta)

    add_dependencies(hdr)

    hdus = fits.HDUList()
    x = fits.PrimaryHDU(qframe.flux.astype('f4'), header=hdr)
    x.header['EXTNAME'] = 'FLUX'
    if units is not None:
        units = str(units)
        if 'BUNIT' in hdr and hdr['BUNIT'] != units:
            log.warning('BUNIT {bunit} != units {units}; using {units}'.format(
                        bunit=hdr['BUNIT'], units=units))
        x.header['BUNIT'] = units
    hdus.append(x)

    hdus.append( fits.ImageHDU(qframe.ivar.astype('f4'), name='IVAR') )
    if qframe.mask is None :
        qframe.mask=np.zeros(qframe.flux.shape,dtype=np.uint32)
    # hdus.append( fits.CompImageHDU(qframe.mask, name='MASK') )
    hdus.append( fits.ImageHDU(qframe.mask, name='MASK') )

    if qframe.sigma is None :
        qframe.sigma=np.zeros(qframe.flux.shape,dtype=np.float32)
    hdus.append( fits.ImageHDU(qframe.sigma.astype('f4'), name='YSIGMA') )

    hdus.append( fits.ImageHDU(qframe.wave.astype('f8'), name='WAVELENGTH') )
    hdus[-1].header['BUNIT'] = 'Angstrom'
    if fibermap is not None:
        fibermap = encode_table(fibermap)  #- unicode -> bytes
        fibermap.meta['EXTNAME'] = 'FIBERMAP'
        hdus.append( fits.convenience.table_to_hdu(fibermap) )
    elif qframe.fibermap is not None:
        fibermap = encode_table(qframe.fibermap)  #- unicode -> bytes
        fibermap.meta['EXTNAME'] = 'FIBERMAP'
        hdus.append( fits.convenience.table_to_hdu(fibermap) )
    elif qframe.spectrograph is not None:
        x.header['FIBERMIN'] = 500*qframe.spectrograph  # Hard-coded (as in desispec.qproc.qframe)
    else:
        log.error("You are likely writing a qframe without sufficient fiber info")
        raise ValueError('no fibermap')

    hdus.writeto(outfile+'.tmp', overwrite=True, checksum=True)
    os.rename(outfile+'.tmp', outfile)

    return outfile

def read_qframe(filename, nspec=None, skip_resolution=False):
    """Reads a frame fits file and returns its data.

    Args:
        filename: path to a file
        skip_resolution: bool, option
            Speed up read time (>5x) by avoiding the Resolution matrix

    Returns:
        desispec.Frame object with attributes wave, flux, ivar, etc.
    """
    log = get_logger()

    if not os.path.isfile(filename) :
        raise IOError("cannot open"+filename)

    fx = fits.open(filename, uint=True, memmap=False)
    hdr = fx[0].header
    flux = native_endian(fx['FLUX'].data.astype('f8'))
    ivar = native_endian(fx['IVAR'].data.astype('f8'))
    wave = native_endian(fx['WAVELENGTH'].data.astype('f8'))

    if wave.shape != flux.shape :
        log.error("{} is not a valid QFrame file because wave.shape != flux.shape".format(filename))
        return None


    if 'MASK' in fx:
        mask = native_endian(fx['MASK'].data)
    else:
        mask = None   #- let the Frame object create the default mask

    if 'SIGMA' in fx:
        sigma = native_endian(fx['SIGMA'].data.astype('f8'))
    else:
        sigma = None


    if 'FIBERMAP' in fx:
        fibermap = fx['FIBERMAP'].data
        fibers   = fibermap['FIBER']
    else:
        fibermap = None
        fibers = None

    fx.close()

    if nspec is not None:
        flux = flux[0:nspec]
        ivar = ivar[0:nspec]
        if mask is not None:
            mask = mask[0:nspec]
        if fibermap is not None:
            fibermap = fibermap[:][0:nspec]
        if fibers is not None:
            fibers = fibers[:][0:nspec]

    # return flux,ivar,wave,resolution_data, hdr
    qframe = QFrame(wave, flux, ivar, mask=mask, sigma=sigma, meta=hdr, fibermap=fibermap, fibers=fibers)

    # Return
    return qframe
