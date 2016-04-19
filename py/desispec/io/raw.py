'''
I/O for DESI raw data files

See DESI-1229 for format details
TODO: move into datamodel after we have verified the format
'''

import os.path
from astropy.io import fits

from desispec.preproc import preproc
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
    rawimage, header = fits.getdata(filename, extname=camera, header=True)
    img = preproc(rawimage, header, **kwargs)
    return img

def write_raw(filename, rawdata, header, camera=None, primary_header=None):
    '''
    Write raw pixel data to a DESI raw data file
    
    Args:
        filename : file name to write data; if this exists, append a new HDU
        rawdata : 2D ndarray of raw pixel data including overscans
        header : dict-like object or fits.Header
    
    Options:
        camera : B0, R1 .. Z9 - override value in header
        primary_header : header to write in HDU0; must be provided if filename
            doesn't yet exist
    
    The primary utility of this function over raw fits calls is to ensure
    that all necessary keywords are present before writing the file
    '''
    #- Check required keywords first
    missing_keywords = list()
    if camera is None and 'CAMERA' not in header:
        log.error("Must provide camera keyword or header['CAMERA']")
        missing_keywords.append('CAMERA')
        
    for amp in ['A', 'B', 'C', 'D']:
        for prefix in ['CCDSEC', 'BIASSEC', 'DATASEC']:
            keyword = prefix+amp
            if keyword not in header:
                log.error('Missing keyword '+keyword)
                missing_keywords.append(keyword)

    if primary_header is not None and 'DATE-OBS' not in primary_header:
        log.error('missing keyword DATE-OBS from primary_header')
        missing_keywords.append('DATE-OBS')

    #- Missing GAINx is warning but not error
    for amp in ['A', 'B', 'C', 'D']:
        keyword = 'GAIN'+amp
        if keyword not in header:
            log.warn('Gain keyword {} missing; using 1.0'.format(keyword))
            header[keyword] = 1.0

    #- Stop if any keywords are missing
    if len(missing_keywords) > 0:
        raise ValueError('missing required keywords {}'.format(missing_keywords))

    #- Set EXTNAME=camera
    if camera is not None:
        extname = camera.upper()
        header['CAMERA'] = extname
    else:
        extname = header['CAMERA']

    header['INHERIT'] = True
        
    #- Actually write or update the file
    if os.path.exists(filename):
        hdus = fits.open(filename, mode='append', memmap=False)
        hdus.append(fits.CompImageHDU(rawdata, header=header, extname=extname))
        hdus.flush()
        hdus.close()
    else:
        hdus = fits.HDUList()
        hdus.append(fits.PrimaryHDU(None, header=primary_header))
        hdus.append(fits.CompImageHDU(rawdata, header=header, extname=extname))
        hdus.writeto(filename)