# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.io.spectra
=====================

I/O routines for working with spectral grouping files.

"""
from __future__ import absolute_import, division, print_function

import os
import re
import warnings
import time

import numpy as np
import astropy.io.fits as fits
from astropy.table import Table

from desiutil.depend import add_dependencies
from desiutil.io import encode_table

from .util import fitsheader, native_endian, add_columns

from .frame import read_frame
from .fibermap import fibermap_comments

from ..spectra import Spectra

def write_spectra(outfile, spec, units=None):
    """
    Write Spectra object to FITS file.

    This places the metadata into the header of the (empty) primary HDU.
    The first extension contains the fibermap, and then HDUs are created for
    the different data arrays for each band.

    Floating point data is converted to 32 bits before writing.

    Args:
        outfile (str): path to write
        spec (Spectra): the object containing the data
        units (str): optional string to use for the BUNIT key of the flux
            HDUs for each band.

    Returns:
        The absolute path to the file that was written.

    """

    outfile = os.path.abspath(outfile)

    # Create the parent directory, if necessary.
    dir, base = os.path.split(outfile)
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    # Create HDUs from the data
    all_hdus = fits.HDUList()

    # metadata goes in empty primary HDU
    hdr = fitsheader(spec.meta)
    add_dependencies(hdr)
    
    all_hdus.append(fits.PrimaryHDU(header=hdr))

    # Next is the fibermap
    fmap = spec.fibermap.copy()
    fmap.meta["EXTNAME"] = "FIBERMAP"
    hdu = fits.convenience.table_to_hdu(fmap)

    # Add comments for fibermap columns.
    for i, colname in enumerate(fmap.dtype.names):
        if colname in fibermap_comments:
            key = "TTYPE{}".format(i+1)
            name = hdu.header[key]
            assert name == colname
            comment = fibermap_comments[name]
            hdu.header[key] = (name, comment)
        else:
            print('Unknown comment for {}'.format(colname))

    all_hdus.append(hdu)

    # Now append the data for all bands

    for band in spec.bands:
        hdu = fits.ImageHDU(name="{}_WAVELENGTH".format(band.upper()))
        hdu.header["BUNIT"] = "Angstrom"
        hdu.data = spec.wave[band].astype("f8")
        all_hdus.append(hdu)

        hdu = fits.ImageHDU(name="{}_FLUX".format(band.upper()))
        if units is None:
            hdu.header["BUNIT"] = "1e-17 erg/(s cm2 Angstrom)"
        else:
            hdu.header["BUNIT"] = units
        hdu.data = spec.flux[band].astype("f4")
        all_hdus.append(hdu)

        hdu = fits.ImageHDU(name="{}_IVAR".format(band.upper()))
        hdu.data = spec.ivar[band].astype("f4")
        all_hdus.append(hdu)

        if spec.mask is not None:
            hdu = fits.CompImageHDU(name="{}_MASK".format(band.upper()))
            hdu.data = spec.mask[band].astype(np.uint32)
            all_hdus.append(hdu)

        if spec.resolution_data is not None:
            hdu = fits.ImageHDU(name="{}_RESOLUTION".format(band.upper()))
            hdu.data = spec.resolution_data[band].astype("f4")
            all_hdus.append(hdu)

        if spec.extra is not None:
            for ex in spec.extra[band].items():
                hdu = fits.ImageHDU(name="{}_{}".format(band.upper(), ex[0]))
                hdu.data = ex[1].astype("f4")
                all_hdus.append(hdu)

    if spec.scores is not None :
        print("writing spec.scores")
        hdu = fits.convenience.table_to_hdu(spec.scores)
        hdu.header["EXTNAME"]="SCORES"
        all_hdus.append(hdu)

    try:
        all_hdus.writeto("{}.tmp".format(outfile), overwrite=True, checksum=True)
    except TypeError:
        all_hdus.writeto("{}.tmp".format(outfile), clobber=True, checksum=True)
    os.rename("{}.tmp".format(outfile), outfile)

    return outfile


def read_spectra(infile, single=False):
    """
    Read Spectra object from FITS file.

    This reads data written by the write_spectra function.  A new Spectra
    object is instantiated and returned.

    Args:
        infile (str): path to read
        single (bool): if True, keep spectra as single precision in memory.

    Returns (Spectra):
        The object containing the data read from disk.

    """

    ftype = np.float64
    if single:
        ftype = np.float32

    infile = os.path.abspath(infile)
    if not os.path.isfile(infile):
        raise IOError("{} is not a file".format(infile))

    hdus = fits.open(infile, mode="readonly")
    nhdu = len(hdus)

    # load the metadata.

    meta = dict(hdus[0].header)

    # initialize data objects

    bands = []
    fmap = None
    wave = None
    flux = None
    ivar = None
    mask = None
    res = None
    extra = None
    scores = None

    # For efficiency, go through the HDUs in disk-order.  Use the
    # extension name to determine where to put the data.  We don't
    # explicitly copy the data, since that will be done when constructing
    # the Spectra object.

    for h in range(1, nhdu):
        name = hdus[h].header["EXTNAME"]
        if name == "FIBERMAP":
            fmap = encode_table(Table(hdus[h].data, copy=True).as_array())
        elif name == "SCORES":
            scores = encode_table(Table(hdus[h].data, copy=True).as_array())
        else:
            # Find the band based on the name
            mat = re.match(r"(.*)_(.*)", name)
            if mat is None:
                raise RuntimeError("FITS extension name {} does not contain the band".format(name))
            band = mat.group(1).lower()
            type = mat.group(2)
            if band not in bands:
                bands.append(band)
            if type == "WAVELENGTH":
                if wave is None:
                    wave = {}
                wave[band] = native_endian(hdus[h].data.astype(ftype))
            elif type == "FLUX":
                if flux is None:
                    flux = {}
                flux[band] = native_endian(hdus[h].data.astype(ftype))
            elif type == "IVAR":
                if ivar is None:
                    ivar = {}
                ivar[band] = native_endian(hdus[h].data.astype(ftype))
            elif type == "MASK":
                if mask is None:
                    mask = {}
                mask[band] = native_endian(hdus[h].data.astype(np.uint32))
            elif type == "RESOLUTION":
                if res is None:
                    res = {}
                res[band] = native_endian(hdus[h].data.astype(ftype))
            else:
                # this must be an "extra" HDU
                if extra is None:
                    extra = {}
                if band not in extra:
                    extra[band] = {}
                extra[band][type] = native_endian(hdus[h].data.astype(ftype))

    # Construct the Spectra object from the data.  If there are any
    # inconsistencies in the sizes of the arrays read from the file,
    # they will be caught by the constructor.

    spec = Spectra(bands, wave, flux, ivar, mask=mask, resolution_data=res,
        fibermap=fmap, meta=meta, extra=extra, single=single, scores=scores)

    hdus.close()

    return spec

def read_frame_as_spectra(filename, night=None, expid=None, band=None, single=False):
    """
    Read a FITS file containing a Frame and return a Spectra.

    A Frame file is very close to a Spectra object (by design), and
    only differs by missing the NIGHT and EXPID in the fibermap, as
    well as containing only one band of data.

    Args:
        infile (str): path to read

    Options:
        night (int): the night value to use for all rows of the fibermap.
        expid (int): the expid value to use for all rows of the fibermap.
        band (str): the name of this band.
        single (bool): if True, keep spectra as single precision in memory.

    Returns (Spectra):
        The object containing the data read from disk.

    """
    fr = read_frame(filename)
    if fr.fibermap is None:
        raise RuntimeError("reading Frame files into Spectra only supported if a fibermap exists")

    nspec = len(fr.fibermap)

    if band is None:
        band = fr.meta['camera'][0]

    if night is None:
        night = fr.meta['night']

    if expid is None:
        expid = fr.meta['expid']

    fmap = np.asarray(fr.fibermap.copy())
    fmap = add_columns(fmap,
                       ['NIGHT', 'EXPID', 'TILEID'],
                       [np.int32(night), np.int32(expid), np.int32(fr.meta['TILEID'])],
                       )

    fmap = encode_table(fmap)

    bands = [ band ]

    mask = None
    if fr.mask is not None:
        mask = {band : fr.mask}

    res = None
    if fr.resolution_data is not None:
        res = {band : fr.resolution_data}

    extra = None
    if fr.chi2pix is not None:
        extra = {band : {"CHI2PIX" : fr.chi2pix}}

    spec = Spectra(bands, {band : fr.wave}, {band : fr.flux}, {band : fr.ivar}, 
        mask=mask, resolution_data=res, fibermap=fmap, meta=fr.meta, 
        extra=extra, single=single, scores=fr.scores)

    return spec


