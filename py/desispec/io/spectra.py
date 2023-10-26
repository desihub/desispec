# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.io.spectra
===================

I/O routines for working with spectral grouping files.

"""
from __future__ import absolute_import, division, print_function

import os
import re
import requests
import warnings
import time
import glob

import numpy as np
import astropy.units as u
import astropy.io.fits as fits
import astropy.table
from astropy.table import Table
import fitsio

from desiutil.depend import add_dependencies
from desiutil.io import encode_table
from desiutil.log import get_logger

from .util import fitsheader, native_endian, add_columns, checkgzip
from .util import get_tempfilename
from . import iotime

from .frame import read_frame
from .fibermap import fibermap_comments

from ..spectra import Spectra, stack
from .meta import specprod_root

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
    log = get_logger()
    outfile = os.path.abspath(outfile)

    # Create the parent directory, if necessary.
    dir, base = os.path.split(outfile)
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Create HDUs from the data
    all_hdus = fits.HDUList()

    # metadata goes in empty primary HDU
    hdr = fitsheader(spec.meta)
    hdr['LONGSTRN'] = 'OGIP 1.0'
    add_dependencies(hdr)

    all_hdus.append(fits.PrimaryHDU(header=hdr))

    # Next is the fibermap
    fmap = encode_table(spec.fibermap.copy())
    fmap.meta['EXTNAME'] = 'FIBERMAP'
    fmap.meta['LONGSTRN'] = 'OGIP 1.0'
    add_dependencies(fmap.meta)

    with warnings.catch_warnings():
        #- nanomaggies aren't an official IAU unit but don't complain
        warnings.filterwarnings('ignore', '.*nanomaggies.*')
        hdu = fits.convenience.table_to_hdu(fmap)

    # Add comments for fibermap columns.
    for i, colname in enumerate(fmap.dtype.names):
        if colname in fibermap_comments:
            key = "TTYPE{}".format(i+1)
            name = hdu.header[key]
            assert name == colname
            comment = fibermap_comments[name]
            hdu.header[key] = (name, comment)

    all_hdus.append(hdu)

    # Optional: exposure-fibermap, used in coadds
    if spec.exp_fibermap is not None:
        expfmap = encode_table(spec.exp_fibermap.copy())
        expfmap.meta["EXTNAME"] = "EXP_FIBERMAP"
        with warnings.catch_warnings():
            #- nanomaggies aren't an official IAU unit but don't complain
            warnings.filterwarnings('ignore', '.*nanomaggies.*')
            hdu = fits.convenience.table_to_hdu(expfmap)

        # Add comments for exp_fibermap columns.
        for i, colname in enumerate(expfmap.dtype.names):
            if colname in fibermap_comments:
                key = "TTYPE{}".format(i+1)
                name = hdu.header[key]
                assert name == colname
                comment = fibermap_comments[name]
                hdu.header[key] = (name, comment)

        all_hdus.append(hdu)

    # Now append the data for all bands

    for band in spec.bands:
        hdu = fits.ImageHDU(name="{}_WAVELENGTH".format(band.upper()))
        hdu.header["BUNIT"] = "Angstrom"
        hdu.data = spec.wave[band].astype("f8")
        all_hdus.append(hdu)

        hdu = fits.ImageHDU(name="{}_FLUX".format(band.upper()))
        if units is None:
            hdu.header["BUNIT"] = "10**-17 erg/(s cm2 Angstrom)"
        else:
            hdu.header["BUNIT"] = units
        hdu.data = spec.flux[band].astype("f4")
        all_hdus.append(hdu)

        hdu = fits.ImageHDU(name="{}_IVAR".format(band.upper()))
        if units is None:
            hdu.header["BUNIT"] = '10**+34 (s2 cm4 Angstrom2) / erg2'
        else:
            hdu.header["BUNIT"] = ((u.Unit(units, format='fits'))**-2).to_string('fits')
        hdu.data = spec.ivar[band].astype("f4")
        all_hdus.append(hdu)

        if spec.mask is not None:
            # hdu = fits.CompImageHDU(name="{}_MASK".format(band.upper()))
            hdu = fits.ImageHDU(name="{}_MASK".format(band.upper()))
            hdu.data = spec.mask[band].astype(np.int32)
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
        scores_tbl = encode_table(spec.scores)  #- unicode -> bytes
        scores_tbl.meta['EXTNAME'] = 'SCORES'
        all_hdus.append( fits.convenience.table_to_hdu(scores_tbl) )
        # add comments in header
        if hasattr(spec, 'scores_comments') and spec.scores_comments is not None:
            hdu=all_hdus['SCORES']
            for i in range(1,999):
                key = 'TTYPE'+str(i)
                if key in hdu.header:
                    value = hdu.header[key]
                    if value in spec.scores_comments.keys() :
                        hdu.header[key] = (value, spec.scores_comments[value])

    if spec.extra_catalog is not None:
        extra_catalog = encode_table(spec.extra_catalog)
        extra_catalog.meta['EXTNAME'] = 'EXTRA_CATALOG'
        all_hdus.append(fits.convenience.table_to_hdu(extra_catalog))

    t0 = time.time()
    tmpfile = get_tempfilename(outfile)
    all_hdus.writeto(tmpfile, overwrite=True, checksum=True)
    os.rename(tmpfile, outfile)
    duration = time.time() - t0
    log.info(iotime.format('write', outfile, duration))

    return outfile

def _read_image(hdus, extname, dtype, rows=None):
    """
    Helper function to read extname from fitsio.FITS hdus, filter by rows,
    convert to native endian, and cast to dtype.  Returns image.
    """
    data = hdus[extname].read()
    if rows is not None:
        data = data[rows]

    return native_endian(data).astype(dtype)


def read_spectra(
    infile,
    single=False,
    targetids=None,
    rows=None,
    skip_hdus=None,
    select_columns={
        "FIBERMAP": None,
        "EXP_FIBERMAP": None,
        "SCORES": None,
        "EXTRA_CATALOG": None,
    },
):
    """
    Read Spectra object from FITS file.

    This reads data written by the write_spectra function.  A new Spectra
    object is instantiated and returned.

    Args:
        infile (str): path to read
        single (bool): if True, keep spectra as single precision in memory.
        targetids (list): Optional, list of targetids to read from file, if present.
        rows (list): Optional, list of rows to read from file
        skip_hdus (list): Optional, list/set/tuple of HDUs to skip
        select_columns (dict): Optional, dictionary to select column names to be read. Default, all columns are read.

    Returns (Spectra):
        The object containing the data read from disk.

    `skip_hdus` options are FIBERMAP, EXP_FIBERMAP, SCORES, EXTRA_CATALOG, MASK, RESOLUTION;
    where MASK and RESOLUTION mean to skip those for all cameras.
    Note that WAVE, FLUX, and IVAR are always required.

    If a table HDU is not listed in `select_columns`, all of its columns will be read

    User can optionally specify targetids OR rows, but not both
    """
    log = get_logger()
    # code to check for file locally, and then recreate file structure and save if not local

    # Check for remote file (AWS)
    ftype = np.float64
    if single:
        ftype = np.float32
    if 'http:' in infile or 'https:' in infile:
        # Save remote image to local file
        img_data = requests.get(infile).content
        file_name = infile.split('/')[-1] # Get file name from url
        with open(file_name, 'wb') as handler:
            handler.write(img_data)
        infile = file_name
    infile = checkgzip(infile)
    infile = os.path.abspath(infile)
    if not os.path.isfile(infile):
        raise IOError("{} is not a file".format(infile))

    t0 = time.time()
    hdus = fitsio.FITS(infile, mode="r")
    nhdu = len(hdus)

    if targetids is not None and rows is not None:
        raise ValueError('Set rows or targetids but not both')

    #- default skip_hdus empty set -> include everything, without
    #- having to check for None before checking if X is in skip_hdus
    if skip_hdus is None:
        skip_hdus = set()

    #- Map targets -> rows and exp_rows.
    #- Note: coadds can have uncoadded EXP_FIBERMAP HDU with more rows than
    #- the coadded FIBERMAP HDU, so track rows vs. exp_rows separately
    exp_rows = None
    if targetids is not None:
        targetids = np.atleast_1d(targetids)
        file_targetids = hdus["FIBERMAP"].read(columns="TARGETID")
        rows = np.where(np.isin(file_targetids, targetids))[0]
        if 'EXP_FIBERMAP' in hdus and 'EXP_FIBERMAP' not in skip_hdus:
            exp_targetids = hdus["EXP_FIBERMAP"].read(columns="TARGETID")
            exp_rows = np.where(np.isin(exp_targetids, targetids))[0]
        if len(rows) == 0:
            return Spectra()
    elif rows is not None:
        rows = np.asarray(rows)
        # figure out exp_rows
        file_targetids = hdus["FIBERMAP"].read(rows=rows, columns="TARGETID")
        if 'EXP_FIBERMAP' in hdus and 'EXP_FIBERMAP' not in skip_hdus:
            exp_targetids = hdus["EXP_FIBERMAP"].read(columns="TARGETID")
            exp_rows = np.where(np.isin(exp_targetids, file_targetids))[0]
        
    if select_columns is None:
        select_columns = dict()

    for extname in ("FIBERMAP", "EXP_FIBERMAP", "SCORES", "EXTRA_CATALOG"):
        if extname not in select_columns:
            select_columns[extname] = None

    # load the metadata.
    meta = dict(hdus[0].read_header())

    # initialize data objects

    bands = []
    fmap = None
    expfmap = None
    wave = None
    flux = None
    ivar = None
    mask = None
    res = None
    extra = None
    extra_catalog = None
    scores = None

    # For efficiency, go through the HDUs in disk-order.  Use the
    # extension name to determine where to put the data.  We don't
    # explicitly copy the data, since that will be done when constructing
    # the Spectra object.

    for h in range(1, nhdu):
        name = hdus[h].read_header()["EXTNAME"]
        log.debug('Reading %s', name)
        if name == "FIBERMAP" and name not in skip_hdus:
            fmap = encode_table(
                Table(
                    hdus[h].read(rows=rows, columns=select_columns["FIBERMAP"]),
                    copy=True,
                ).as_array()
            )
        elif name == "EXP_FIBERMAP" and name not in skip_hdus:
            expfmap = encode_table(
                Table(
                    hdus[h].read(rows=exp_rows, columns=select_columns["EXP_FIBERMAP"]),
                    copy=True,
                ).as_array()
            )
        elif name == "SCORES" and name not in skip_hdus:
            scores = encode_table(
                Table(
                    hdus[h].read(rows=rows, columns=select_columns["SCORES"]),
                    copy=True,
                ).as_array()
            )
        elif name == "EXTRA_CATALOG" and name not in skip_hdus:
            extra_catalog = encode_table(
                Table(
                    hdus[h].read(
                        rows=rows, columns=select_columns["EXTRA_CATALOG"]
                    ),
                    copy=True,
                ).as_array()
            )
        else:
            # Find the band based on the name
            mat = re.match(r"(.*)_(.*)", name)
            if mat is None:
                raise RuntimeError(
                    "FITS extension name {} does not contain the band".format(name)
                )
            band = mat.group(1).lower()
            type = mat.group(2)
            if band not in bands:
                bands.append(band)
            if type == "WAVELENGTH":
                if wave is None:
                    wave = {}
                # - Note: keep original float64 resolution for wavelength
                wave[band] = native_endian(hdus[h].read())
            elif type == "FLUX":
                if flux is None:
                    flux = {}
                flux[band] = _read_image(hdus, h, ftype, rows=rows)
            elif type == "IVAR":
                if ivar is None:
                    ivar = {}
                ivar[band] = _read_image(hdus, h, ftype, rows=rows)
            elif type == "MASK" and type not in skip_hdus:
                if mask is None:
                    mask = {}
                mask[band] = _read_image(hdus, h, np.uint32, rows=rows)
            elif type == "RESOLUTION" and type not in skip_hdus:
                if res is None:
                    res = {}
                res[band] = _read_image(hdus, h, ftype, rows=rows)
            elif type != "MASK" and type != "RESOLUTION" and type not in skip_hdus:
                # this must be an "extra" HDU
                if extra is None:
                    extra = {}
                if band not in extra:
                    extra[band] = {}

                extra[band][type] = _read_image(hdus, h, ftype, rows=rows)

    hdus.close()
    duration = time.time() - t0
    log.info(iotime.format("read", infile, duration))

    # Construct the Spectra object from the data.  If there are any
    # inconsistencies in the sizes of the arrays read from the file,
    # they will be caught by the constructor.

    spec = Spectra(
        bands,
        wave,
        flux,
        ivar,
        mask=mask,
        resolution_data=res,
        fibermap=fmap,
        exp_fibermap=expfmap,
        meta=meta,
        extra=extra,
        extra_catalog=extra_catalog,
        single=single,
        scores=scores,
    )

    # if needed, sort spectra to match order of targetids, which could be
    # different than the order they appear in the file
    if targetids is not None:
        from desispec.util import ordered_unique
        #- Input targetids that we found in the file, in the order they appear in targetids
        ii = np.isin(targetids, spec.fibermap['TARGETID'])
        found_targetids = ordered_unique(targetids[ii])
        log.debug('found_targetids=%s', found_targetids)

        #- Unique targetids of input file in the order they first appear
        input_targetids = ordered_unique(spec.fibermap['TARGETID'])
        log.debug('input_targetids=%s', np.asarray(input_targetids))

        #- Only reorder if needed
        if not np.all(input_targetids == found_targetids):
            rows = np.concatenate([np.where(spec.fibermap['TARGETID'] == tid)[0] for tid in targetids])
            log.debug("spec.fibermap['TARGETID'] = %s", np.asarray(spec.fibermap['TARGETID']))
            log.debug("rows for subselection=%s", rows)
            spec = spec[rows]

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
    filename = checkgzip(filename)
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

def read_tile_spectra(tileid, night, specprod=None, reduxdir=None, coadd=False,
                      single=False, targets=None, fibers=None, redrock=True,
                      group=None):
    """
    Read and return combined spectra for a tile/night

    Args:
        tileid (int) : Tile ID
        night (int or str) : YEARMMDD night or tile group, e.g. 'deep' or 'all'

    Options:
        specprod (str) : overrides $SPECPROD
        reduxdir (str) : overrides $DESI_SPECTRO_REDUX/$SPECPROD
        coadd (bool) : if True, read coadds instead of per-exp spectra
        single (bool) : if True, use float32 instead of double precision
        targets (array-like) : filter by TARGETID
        fibers (array-like) : filter by FIBER
        redrock (bool) : if True, also return row-matched redrock redshift catalog
        group (str) : reads spectra in group (pernight, cumulative, ...)

    Returns: spectra or (spectra, redrock)
        combined Spectra obj for all matching targets/fibers filter
        row-matched redrock catalog (if redrock=True)

    Raises:
        ValueError if no files or matching spectra are found

    Note: the returned spectra are not necessarily in the same order as
    the `targets` or `fibers` input filters
    """
    log = get_logger()
    if reduxdir is None:
        #- will automatically use $SPECPROD if specprod=None
        reduxdir = specprod_root(specprod)

    tiledir = os.path.join(reduxdir, 'tiles')
    nightstr = str(night)
    if group is not None:
        tiledir = os.path.join(tiledir, group)
        if group == 'cumulative':
            nightstr = 'thru'+nightstr

    tiledir = os.path.join(tiledir, str(tileid), str(night))

    if coadd:
        log.debug(f'Reading coadds from {tiledir}')
        prefix = 'coadd'
    else:
        log.debug(f'Reading spectra from {tiledir}')
        prefix = 'spectra'

    specfiles = glob.glob(f'{tiledir}/{prefix}-?-{tileid}-{nightstr}.fits*')

    if len(specfiles) == 0:
        raise ValueError(f'No spectra found in {tiledir}')

    specfiles = sorted(specfiles)

    spectra = list()
    redshifts = list()
    for filename in specfiles:
        log.debug(f'reading {os.path.basename(filename)}')
        sp = read_spectra(filename, single=single)
        if targets is not None:
            keep = np.in1d(sp.fibermap['TARGETID'], targets)
            sp = sp[keep]
        if fibers is not None:
            keep = np.in1d(sp.fibermap['FIBER'], fibers)
            sp = sp[keep]

        if sp.num_spectra() > 0:
            spectra.append(sp)
            if redrock:
                #- Read matching redrock file for this spectra/coadd file
                rrfile = os.path.basename(filename).replace(prefix, 'redrock', 1)
                log.debug(f'Reading {rrfile}')
                rrfile = os.path.join(tiledir, rrfile)
                rr = Table.read(rrfile, 'REDSHIFTS')

                #- Trim rr to only have TARGETIDs in filtered spectra sp
                keep = np.in1d(rr['TARGETID'], sp.fibermap['TARGETID'])
                rr = rr[keep]

                #- spectra files can have multiple entries per TARGETID,
                #- while redrock files have only 1.  Expand to match spectra.
                #- Note: astropy.table.join changes the order
                if len(sp.fibermap) > len(rr):
                    rrx = Table()
                    rrx['TARGETID'] = sp.fibermap['TARGETID']
                    rrx = astropy.table.join(rrx, rr, keys='TARGETID')
                else:
                    rrx = rr

                #- Sort the rrx Table to match the order of sp['TARGETID']
                ii = np.argsort(sp.fibermap['TARGETID'])
                jj = np.argsort(rrx['TARGETID'])
                kk = np.argsort(ii[jj])
                rrx = rrx[kk]

                #- Confirm that we got all that expanding and sorting correct
                assert np.all(sp.fibermap['TARGETID'] == rrx['TARGETID'])
                redshifts.append(rrx)

    if len(spectra) == 0:
        raise ValueError('No spectra found matching filters')

    spectra = stack(spectra)

    if redrock:
        redshifts = astropy.table.vstack(redshifts)
        assert np.all(spectra.fibermap['TARGETID'] == redshifts['TARGETID'])
        return (spectra, redshifts)
    else:
        return spectra
