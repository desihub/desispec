'''
I/O for DESI raw data files

See DESI-1229 for format details
TODO: move into datamodel after we have verified the format
'''

import os.path
from astropy.io import fits
import numpy as np

import desispec.io.util
import desispec.preproc
from desispec.log import get_logger
log = get_logger()

def read_raw(filename, camera, **kwargs):
    '''
    Returns preprocessed raw data from `camera` extension of `filename`

    Args:
        filename : input fits filename with DESI raw data
        camera : camera name (B0,R1, .. Z9) or FITS extension name or number

    Options:
        Other keyword arguments are passed to desispec.preproc.preproc(),
        e.g. bias, pixflat, mask.  See preproc() documentation for details.

    Returns Image object with member variables pix, ivar, mask, readnoise
    '''
    fx = fits.open(filename, memmap=False)
    if camera.upper() not in fx:
        raise IOError('Camera {} not in {}'.format(camera, filename))

    rawimage = fx[camera.upper()].data
    header = fx[camera.upper()].header

    if 'INHERIT' in header and header['INHERIT']:
        h0 = fx[0].header
        for key in h0.keys():
            if key not in header:
                header[key] = h0[key]

    fx.close()

    img = desispec.preproc.preproc(rawimage, header, **kwargs)
    return img

def write_raw(filename, rawdata, header, camera=None, primary_header=None):
    '''
    Write raw pixel data to a DESI raw data file

    Args:
        filename : file name to write data; if this exists, append a new HDU
        rawdata : 2D ndarray of raw pixel data including overscans
        header : dict-like object or fits.Header with keywords
            CCDSECx, BIASSECx, DATASECx where x=1,2,3, or 4

    Options:
        camera : B0, R1 .. Z9 - override value in header
        primary_header : header to write in HDU0 if filename doesn't yet exist

    The primary utility of this function over raw fits calls is to ensure
    that all necessary keywords are present before writing the file.
    CCDSECx, BIASSECx, DATASECx where x=1,2,3, or 4
    DATE-OBS, GAINx and RDNOISEx will generate a non-fatal warning if missing
    '''
    header = desispec.io.util.fitsheader(header)
    primary_header = desispec.io.util.fitsheader(primary_header)

    if rawdata.dtype not in (np.int16, np.int32, np.int64):
        message = 'dtype {} not supported for raw data'.format(rawdata.dtype)
        log.fatal(message)
        raise ValueError(message)

    #- Check required keywords before writing anything
    missing_keywords = list()
    if camera is None and 'CAMERA' not in header:
        log.error("Must provide camera keyword or header['CAMERA']")
        missing_keywords.append('CAMERA')

    for amp in ['1', '2', '3', '4']:
        for prefix in ['CCDSEC', 'BIASSEC', 'DATASEC']:
            keyword = prefix+amp
            if keyword not in header:
                log.error('Missing keyword '+keyword)
                missing_keywords.append(keyword)

    #- Missing DATE-OBS is warning but not error
    if 'DATE-OBS' not in primary_header:
        if 'DATE-OBS' in header:
            primary_header['DATE-OBS'] = header['DATE-OBS']
        else:
            log.warning('missing keyword DATE-OBS')

    #- Missing GAINx is warning but not error
    for amp in ['1', '2', '3', '4']:
        keyword = 'GAIN'+amp
        if keyword not in header:
            log.warn('Gain keyword {} missing; using 1.0'.format(keyword))
            header[keyword] = 1.0

    #- Missing RDNOISEx is warning but not error
    for amp in ['1', '2', '3', '4']:
        keyword = 'RDNOISE'+amp
        if keyword not in header:
            log.warn('Readnoise keyword {} missing'.format(keyword))

    #- Stop if any keywords are missing
    if len(missing_keywords) > 0:
        raise KeyError('missing required keywords {}'.format(missing_keywords))

    #- Set EXTNAME=camera
    if camera is not None:
        header['CAMERA'] = camera
        extname = camera.upper()
    else:
        extname = header['CAMERA'].upper()

    header['INHERIT'] = True

    #- fits.CompImageHDU doesn't know how to fill in default keywords, so
    #- temporarily generate an uncompressed HDU to get those keywords
    header = fits.ImageHDU(rawdata, header=header, name=extname).header

    #- Bizarrely, compression of 64-bit integers isn't supported.
    #- downcast to 32-bit if that won't lose precision.
    #- Real raw data should be 32-bit or 16-bit anyway
    if rawdata.dtype == np.int64:
        if np.max(np.abs(rawdata)) < 2**31:
            rawdata = rawdata.astype(np.int32)

    if rawdata.dtype in (np.int16, np.int32):
        dataHDU = fits.CompImageHDU(rawdata, header=header, name=extname)
    elif rawdata.dtype == np.int64:
        log.warn('Image compression not supported for 64-bit; writing uncompressed')
        dataHDU = fits.ImageHDU(rawdata, header=header, name=extname)
    else:
        log.error("How did we get this far with rawdata dtype {}?".format(rawdata.dtype))
        dataHDU = fits.ImageHDU(rawdata, header=header, name=extname)

    #- Actually write or update the file
    if os.path.exists(filename):
        hdus = fits.open(filename, mode='append', memmap=False)
        if extname in hdus:
            hdus.close()
            raise ValueError('Camera {} already in {}'.format(camera, filename))
        else:
            hdus.append(dataHDU)
            hdus.flush()
            hdus.close()
    else:
        hdus = fits.HDUList()
        hdus.append(fits.PrimaryHDU(None, header=primary_header))
        hdus.append(dataHDU)
        hdus.writeto(filename)
