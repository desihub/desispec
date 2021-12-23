# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.io.fiberflat
=====================

IO routines for fiberflat.
"""
from __future__ import absolute_import
# The line above will help with 2to3 support.
import os
import time
import numpy as np
from astropy.io import fits

from desiutil.io import encode_table
from desiutil.depend import add_dependencies
from desiutil.log import get_logger

from ..fiberflat import FiberFlat
from .meta import findfile
from .util import fitsheader, native_endian, makepath
from . import iotime

def write_fiberflat(outfile,fiberflat,header=None, fibermap=None):
    """Write fiberflat object to outfile

    Args:
        outfile: filepath string or (night, expid, camera) tuple
        fiberflat: FiberFlat object

    Optional:
        header: dict or fits.Header object to use as HDU 0 header
        fibermap: table to store as FIBERMAP HDU

    Returns:
        filepath of file that was written
    """
    log = get_logger()
    outfile = makepath(outfile, 'fiberflat')

    if header is None:
        hdr = fitsheader(fiberflat.header)
    else:
        hdr = fitsheader(header)
    if fiberflat.chi2pdf is not None:
        hdr['chi2pdf'] = float(fiberflat.chi2pdf)

    hdr['EXTNAME'] = 'FIBERFLAT'
    if 'BUNIT' in hdr:
        del hdr['BUNIT']

    add_dependencies(hdr)

    ff = fiberflat   #- shorthand

    hdus = fits.HDUList()
    hdus.append(fits.PrimaryHDU(ff.fiberflat.astype('f4'), header=hdr))
    hdus.append(fits.ImageHDU(ff.ivar.astype('f4'),     name='IVAR'))
    hdus.append(fits.ImageHDU(ff.mask,              name='MASK'))
    hdus.append(fits.ImageHDU(ff.meanspec.astype('f4'), name='MEANSPEC'))
    hdus.append(fits.ImageHDU(ff.wave.astype('f4'),     name='WAVELENGTH'))
    if fibermap is None :
        fibermap=ff.fibermap
    if fibermap is not None:
        fibermap = encode_table(fibermap)  #- unicode -> bytes
        fibermap.meta['EXTNAME'] = 'FIBERMAP'
        hdus.append( fits.convenience.table_to_hdu(fibermap) )
    hdus[0].header['BUNIT'] = ("","adimensional quantity to divide to flatfield a frame")
    hdus["IVAR"].header['BUNIT'] = ("","inverse variance, adimensional")
    hdus["MEANSPEC"].header['BUNIT'] = ("electron/Angstrom")
    hdus["WAVELENGTH"].header['BUNIT'] = 'Angstrom'

    t0 = time.time()
    hdus.writeto(outfile+'.tmp', overwrite=True, checksum=True)
    os.rename(outfile+'.tmp', outfile)
    duration = time.time() - t0
    log.info(iotime.format('write', outfile, duration))

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

    log = get_logger()
    t0 = time.time()
    with fits.open(filename, uint=True, memmap=False) as fx:
        header    = fx[0].header
        fiberflat = native_endian(fx[0].data.astype('f8'))
        ivar      = native_endian(fx["IVAR"].data.astype('f8'))
        mask      = native_endian(fx["MASK"].data)
        meanspec  = native_endian(fx["MEANSPEC"].data.astype('f8'))
        wave      = native_endian(fx["WAVELENGTH"].data.astype('f8'))
        if 'FIBERMAP' in fx:
            fibermap = fx['FIBERMAP'].data
        else:
            fibermap = None

    duration = time.time() - t0
    log.info(iotime.format('read', filename, duration))

    return FiberFlat(wave, fiberflat, ivar, mask, meanspec, header=header, fibermap=fibermap)
