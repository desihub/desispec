"""
desispec.io.frame
=================

I/O routines for Frame objects
"""
import os.path
import time

import numpy as np
import scipy, scipy.sparse
import fitsio
from astropy.io import fits
from astropy.table import Table
import warnings

from desiutil.depend import add_dependencies
from desiutil.log import get_logger

from ..frame import Frame
from .fibermap import read_fibermap, annotate_fibermap
from .meta import findfile, get_nights, get_exposures
from .util import fitsheader, native_endian, makepath, checkgzip
from .util import get_tempfilename
from . import iotime

def write_frame(outfile, frame, header=None, fibermap=None, units=None):
    """Write a frame fits file and returns path to file written.

    Args:
        outfile: full path to output file, or tuple (night, expid, channel)
        frame:  desispec.frame.Frame object with wave, flux, ivar...

    Optional:
        header: astropy.io.fits.Header or dict to override frame.header
        fibermap: table to store as FIBERMAP HDU

    Returns:
        full filepath of output file that was written

    Note:
        to create a Frame object to pass into write_frame,
        frame = Frame(wave, flux, ivar, resolution_data)
    """
    log = get_logger()
    outfile = makepath(outfile, 'frame')

    #- Ignore some known and harmless units warnings
    import warnings
    warnings.filterwarnings('ignore', message="'.*nanomaggies.* did not parse as fits unit.*")
    warnings.filterwarnings('ignore', message=r".*'10\*\*6 arcsec.* did not parse as fits unit.*")

    if header is not None:
        hdr = fitsheader(header)
    else:
        hdr = fitsheader(frame.meta)

    add_dependencies(hdr)

    # Vet
    diagnosis = frame.vet()
    if diagnosis != 0:
        raise IOError("Frame did not pass simple vetting test. diagnosis={:d}".format(diagnosis))

    hdus = fits.HDUList()
    x = fits.PrimaryHDU(frame.flux.astype('f4'), header=hdr)
    x.header['EXTNAME'] = 'FLUX'
    if units is not None:
        units = str(units)
        if 'BUNIT' in hdr and hdr['BUNIT'] != units:
            log.warning('BUNIT {bunit} != units {units}; using {units}'.format(
                        bunit=hdr['BUNIT'], units=units))
        x.header['BUNIT'] = units
    hdus.append(x)

    hdus.append( fits.ImageHDU(frame.ivar.astype('f4'), name='IVAR') )
    # hdus.append( fits.CompImageHDU(frame.mask, name='MASK') )
    hdus.append( fits.ImageHDU(frame.mask, name='MASK') )
    hdus.append( fits.ImageHDU(frame.wave.astype('f8'), name='WAVELENGTH') )
    hdus[-1].header['BUNIT'] = 'Angstrom'
    if frame.resolution_data is not None:
        hdus.append( fits.ImageHDU(frame.resolution_data.astype('f4'), name='RESOLUTION' ) )
    elif frame.wsigma is not None:
        log.debug("Using ysigma from qproc")
        qrimg=fits.ImageHDU(frame.wsigma.astype('f4'), name='YSIGMA' )
        qrimg.header["NDIAG"] =frame.ndiag
        hdus.append(qrimg)
    if fibermap is not None:
        fibermap = Table(fibermap)
        fibermap.meta['EXTNAME'] = 'FIBERMAP'
        add_dependencies(fibermap.meta)
        hdus.append( fits.convenience.table_to_hdu(fibermap) )
        fmhdu = fits.convenience.table_to_hdu(fibermap)
        # TODO: determine provenance: frame, sframe or cframe?
        # TODO: Only cframe has an extra fibermap column.
        fmhdu = annotate_fibermap(fmhdu, survey=fmhdu.header['SURVEY'])
        hdus.append( fmhdu )
    elif frame.fibermap is not None:
        fibermap = Table(frame.fibermap)
        fibermap.meta['EXTNAME'] = 'FIBERMAP'
        fmhdu = fits.convenience.table_to_hdu(fibermap)
        fmhdu = annotate_fibermap(fmhdu, survey=fmhdu.header['SURVEY'])
        hdus.append( fmhdu )
    elif frame.spectrograph is not None:
        x.header['FIBERMIN'] = 500*frame.spectrograph  # Hard-coded (as in desispec.frame)
    else:
        log.error("You are likely writing a frame without sufficient fiber info")

    if frame.chi2pix is not None:
        hdus.append( fits.ImageHDU(frame.chi2pix.astype('f4'), name='CHI2PIX' ) )

    if frame.scores is not None :
        scores_tbl = Table(frame.scores)
        scores_tbl.meta['EXTNAME'] = 'SCORES'
        hdus.append( fits.convenience.table_to_hdu(scores_tbl) )
        if frame.scores_comments is not None : # add comments in header
            hdu=hdus['SCORES']
            for i in range(1,999):
                key = 'TTYPE'+str(i)
                if key in hdu.header:
                    value = hdu.header[key]
                    if value in frame.scores_comments.keys() :
                        hdu.header[key] = (value, frame.scores_comments[value])

    t0 = time.time()
    tmpfile = get_tempfilename(outfile)
    hdus.writeto(tmpfile, overwrite=True, checksum=True)
    os.rename(tmpfile, outfile)
    duration = time.time() - t0
    log.info(iotime.format('write', outfile, duration))

    return outfile


def read_meta_frame(filename, extname=0):
    """ Load the meta information of a Frame
    Args:
        filename: path to a file
        extname: int, optional;  Extension for grabbing header info

    Returns:
        meta: dict or astropy.fits.header

    """
    filename = checkgzip(filename)
    with fits.open(filename, uint=True, memmap=False) as fx:
        hdr = fx[extname].header
    return hdr


def read_frame(filename, nspec=None, skip_resolution=False):
    """Reads a frame fits file and returns its data.

    Args:
        filename: path to a file, or (night, expid, camera) tuple where
            night = string YEARMMDD
            expid = integer exposure ID
            camera = b0, r1, .. z9
        skip_resolution: bool, option
            Speed up read time (>5x) by avoiding the Resolution matrix

    Returns:
        desispec.Frame object with attributes wave, flux, ivar, etc.
    """
    log = get_logger()

    #- check if filename is (night, expid, camera) tuple instead
    if not isinstance(filename, str):
        night, expid, camera = filename
        filename = findfile('frame', night, expid, camera)

    #- check for gzip, raise FileNotFoundError if neither exists
    filename = checkgzip(filename)

    t0 = time.time()
    fx = fitsio.FITS(filename)
    hdr = fx[0].read_header()
    flux = native_endian(fx['FLUX'].read().astype('f8'))
    ivar = native_endian(fx['IVAR'].read().astype('f8'))
    wave = native_endian(fx['WAVELENGTH'].read().astype('f8'))
    if 'MASK' in fx:
        mask = native_endian(fx['MASK'].read().astype(np.uint32))
    else:
        mask = None   #- let the Frame object create the default mask

    # Init
    resolution_data=None
    qwsigma=None
    qndiag=None
    fibermap = None
    chi2pix = None
    scores = None
    scores_comments = None

    if skip_resolution:
        pass
    elif 'RESOLUTION' in fx:
        resolution_data = native_endian(fx['RESOLUTION'].read().astype('f8'))
    elif 'QUICKRESOLUTION' in fx:
        qr=fx['QUICKRESOLUTION'].header
        qndiag =qr['NDIAG']
        qwsigma=native_endian(fx['QUICKRESOLUTION'].read().astype('f4'))

    if 'FIBERMAP' in fx:
        fibermap = read_fibermap(fx)
    else:
        fibermap = None

    if 'CHI2PIX' in fx:
        chi2pix = native_endian(fx['CHI2PIX'].read().astype('f8'))
    else:
        chi2pix = None

    if 'SCORES' in fx:
        scores = fx['SCORES'].read()
        # I need to open the header to read the comments
        scores_comments = dict()
        head   = fx['SCORES'].read_header()
        for i in range(1,len(scores.dtype.names)+1) :
            k='TTYPE'+str(i)
            scores_comments[head[k]]=head.get_comment(k)
    else:
        scores = None
        scores_comments = None

    fx.close()
    duration = time.time() - t0
    log.info(iotime.format('read', filename, duration))

    if nspec is not None:
        flux = flux[0:nspec]
        ivar = ivar[0:nspec]
        if resolution_data is not None:
            resolution_data = resolution_data[0:nspec]
        else:
            qwsigma=qwsigma[0:nspec]
        if chi2pix is not None:
            chi2pix = chi2pix[0:nspec]
        if mask is not None:
            mask = mask[0:nspec]

    if 'SPECGRPH' in hdr:
        spectrograph = hdr['SPECGRPH']
    elif 'CAMERA' in hdr:
        spectrograph = int(hdr['CAMERA'][1])
    else:
        spectrograph = None

    # return flux,ivar,wave,resolution_data, hdr
    frame = Frame(wave, flux, ivar, mask, resolution_data, meta=hdr, fibermap=fibermap, chi2pix=chi2pix,
                  scores=scores,scores_comments=scores_comments,
                  wsigma=qwsigma,ndiag=qndiag, suppress_res_warning=skip_resolution,
                  spectrograph=spectrograph)

    # This Frame came from a file, so set that
    frame.filename = os.path.abspath(filename)

    # Vette
    diagnosis = frame.vet()
    if diagnosis != 0:
        warnings.warn("Frame did not pass simple vetting test. diagnosis={:d}".format(diagnosis))
        log.error("Frame did not pass simple vetting test. diagnosis={:d}".format(diagnosis))
    # Return
    return frame

def search_for_framefile(frame_file, specprod_dir=None):
    """ Search for an input frame_file in the desispec redux hierarchy
    Args:
        frame_file:  str
        specprod_dir: str, optional

    Returns:
        mfile: str,  full path to frame_file if found else raise error
    """
    log=get_logger()
    # Parse frame file
    path, ifile = os.path.split(frame_file)
    splits = ifile.split('-')
    root = splits[0]
    camera = splits[1]
    fexposure = int(splits[2].split('.')[0])

    # Loop on nights
    nights = get_nights(specprod_dir=specprod_dir)
    for night in nights:
        for exposure in get_exposures(night, specprod_dir=specprod_dir):
            if exposure == fexposure:
                mfile = findfile(root, camera=camera, night=night, expid=exposure, specprod_dir=specprod_dir)
                if os.path.isfile(mfile):
                    return mfile
                else:
                    log.error("Expected file {:s} not found..".format(mfile))
