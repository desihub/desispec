"""
desispec.io.frame
=================

I/O routines for Frame objects
"""
import os.path

import numpy as np
import scipy, scipy.sparse
from astropy.io import fits
import warnings

from desiutil.depend import add_dependencies
from desiutil.io import encode_table

from ..frame import Frame
from .meta import findfile, get_nights, get_exposures
from .util import fitsheader, native_endian, makepath
from desiutil.log import get_logger

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

    if header is not None:
        hdr = fitsheader(header)
    else:
        hdr = fitsheader(frame.meta)

    add_dependencies(hdr)

    # Vette
    diagnosis = frame.vet()
    if diagnosis != 0:
        raise IOError("Frame did not pass simple vetting test. diagnosis={:d}".format(diagnosis))

    hdus = fits.HDUList()
    x = fits.PrimaryHDU(frame.flux.astype('f4'), header=hdr)
    x.header['EXTNAME'] = 'FLUX'
    if units is not None:
        units = str(units)
        if 'BUNIT' in hdr and hdr['BUNIT'] != units:
            log.warn('BUNIT {bunit} != units {units}; using {units}'.format(
                    bunit=hdr['BUNIT'], units=units))
        x.header['BUNIT'] = units
    hdus.append(x)

    hdus.append( fits.ImageHDU(frame.ivar.astype('f4'), name='IVAR') )
    hdus.append( fits.CompImageHDU(frame.mask, name='MASK') )
    hdus.append( fits.ImageHDU(frame.wave.astype('f4'), name='WAVELENGTH') )
    hdus[-1].header['BUNIT'] = 'Angstrom'
    if frame.resolution_data is not None:
        hdus.append( fits.ImageHDU(frame.resolution_data.astype('f4'), name='RESOLUTION' ) )
    elif frame.coeffs is not None:
        log.info("SAMI SAMI SAMI")
        qrimg=fits.ImageHDU(frame.coeffs.astype('f8'), name='QUICKRESOLUTION' ) 
        qrimg.header["WMIN"]  =frame.wmin
        qrimg.header["WMAX"]  =frame.wmax
        qrimg.header["YMIN"]  =frame.ymin
        qrimg.header["YMAX"]  =frame.ymax
        qrimg.header["NPIX_Y"]=frame.npix_y
        qrimg.header["NDIAG"] =frame.ndiag
        hdus.append(qrimg)
    if fibermap is not None:
        fibermap = encode_table(fibermap)  #- unicode -> bytes
        fibermap.meta['EXTNAME'] = 'FIBERMAP'
        hdus.append( fits.convenience.table_to_hdu(fibermap) )
    elif frame.fibermap is not None:
        fibermap = encode_table(frame.fibermap)  #- unicode -> bytes
        fibermap.meta['EXTNAME'] = 'FIBERMAP'
        hdus.append( fits.convenience.table_to_hdu(fibermap) )
    elif frame.spectrograph is not None:
        x.header['FIBERMIN'] = 500*frame.spectrograph  # Hard-coded (as in desispec.frame)
    else:
        log.error("You are likely writing a frame without sufficient fiber info")

    if frame.chi2pix is not None:
        hdus.append( fits.ImageHDU(frame.chi2pix.astype('f4'), name='CHI2PIX' ) )

    hdus.writeto(outfile+'.tmp', clobber=True, checksum=True)
    os.rename(outfile+'.tmp', outfile)

    return outfile


def read_meta_frame(filename, extname=0):
    """ Load the meta information of a Frame
    Args:
        filename: path to a file
        extname: int, optional;  Extension for grabbing header info

    Returns:
        meta: dict or astropy.fits.header

    """
    fx = fits.open(filename, uint=True, memmap=False)
    hdr = fx[extname].header
    return hdr


def read_frame(filename, nspec=None):
    """Reads a frame fits file and returns its data.

    Args:
        filename: path to a file, or (night, expid, camera) tuple where
            night = string YEARMMDD
            expid = integer exposure ID
            camera = b0, r1, .. z9

    Returns:
        desispec.Frame object with attributes wave, flux, ivar, etc.
    """
    log = get_logger()

    #- check if filename is (night, expid, camera) tuple instead
    if not isinstance(filename, str):
        night, expid, camera = filename
        filename = findfile('frame', night, expid, camera)

    if not os.path.isfile(filename) :
        raise IOError("cannot open"+filename)

    fx = fits.open(filename, uint=True, memmap=False)
    hdr = fx[0].header
    flux = native_endian(fx['FLUX'].data.astype('f8'))
    ivar = native_endian(fx['IVAR'].data.astype('f8'))
    wave = native_endian(fx['WAVELENGTH'].data.astype('f8'))
    if 'MASK' in fx:
        mask = native_endian(fx['MASK'].data)
    else:
        mask = None   #- let the Frame object create the default mask

    resolution_data=None
    qwcoeff=None
    qwmin=None
    qwmax=None
    qymin=None
    qymax=None
    qndiag=None
    qnpix_y=None
    if 'RESOLUTION' in fx:
        resolution_data = native_endian(fx['RESOLUTION'].data.astype('f8'))
    elif 'QUICKRESOLUTION' in fx:
        qr=fx['QUICKRESOLUTION'].header
        qwmin  =native_endian(qr['WMIN'])
        qwmax  =native_endian(qr['WMAX'])
        qymin  =native_endian(qr['YMIN'])
        qymax  =native_endian(qr['YMAX'])
        qnpix_y=native_endian(qr['NPIX_Y'])
        qndiag =native_endian(qr['NDIAG'])
        qwcoeff=native_endiag(fx['QUICKRESOLUTION'].data.astype('f8'))
        
    if 'FIBERMAP' in fx:
        fibermap = fx['FIBERMAP'].data
    else:
        fibermap = None

    if 'CHI2PIX' in fx:
        chi2pix = native_endian(fx['CHI2PIX'].data.astype('f8'))
    else:
        chi2pix = None

    fx.close()

    if nspec is not None:
        flux = flux[0:nspec]
        ivar = ivar[0:nspec]
        if resolution_data is not None:
            resolution_data = resolution_data[0:nspec]
        else:
            qwcoeff=qwcoeff[0:nspec]
        if chi2pix is not None:
            chi2pix = chi2pix[0:nspec]
        if mask is not None:
            mask = mask[0:nspec]

    # return flux,ivar,wave,resolution_data, hdr
    frame = Frame(wave, flux, ivar, mask, resolution_data, meta=hdr, fibermap=fibermap, chi2pix=chi2pix,
                  coefficients=qwcoeff,ndiag=qndiag,ymin=qymin,ymax=qymax,wmin=qwmin,wmax=qwmax,npix_y=qnpix_y
    )

    # Vette
    diagnosis = frame.vet()
    if diagnosis != 0:
        warnings.warn("Frame did not pass simple vetting test. diagnosis={:d}".format(diagnosis))
        log.error("Frame did not pass simple vetting test. diagnosis={:d}".format(diagnosis))
    # Return
    return frame


def search_for_framefile(frame_file):
    """ Search for an input frame_file in the desispec redux hierarchy
    Args:
        frame_file:  str

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
    nights = get_nights()
    for night in nights:
        for exposure in get_exposures(night):
            if exposure == fexposure:
                mfile = findfile(root, camera=camera, night=night, expid=exposure)
                if os.path.isfile(mfile):
                    return mfile
                else:
                    log.error("Expected file {:s} not found..".format(mfile))
