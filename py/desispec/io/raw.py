'''
I/O for DESI raw data files

See DESI-1229 for format details
TODO: move into datamodel after we have verified the format
'''

import os.path
from astropy.io import fits
import numpy as np

from desiutil.depend import add_dependencies

import desispec.io.util
import desispec.preproc
from desiutil.log import get_logger

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
    
    log = get_logger()
    
    fx = fits.open(filename, memmap=False)
    if camera.upper() not in fx:
        raise IOError('Camera {} not in {}'.format(camera, filename))

    rawimage = fx[camera.upper()].data
    header = fx[camera.upper()].header
    hdu=0
    while True :
        primary_header= fx[hdu].header
        if "EXPTIME" in primary_header : break

        if len(fx)>hdu+1 :
            log.warning("Did not find header keyword EXPTIME in hdu {}, moving to the next".format(hdu))
            hdu +=1 
        else :
            log.error("Did not find header keyword EXPTIME in any HDU of {}".format(filename))
            raise KeyError("Did not find header keyword EXPTIME in any HDU of {}".format(filename))
    
    blacklist = ["EXTEND","SIMPLE","NAXIS1","NAXIS2","CHECKSUM","DATASUM","XTENSION","EXTNAME","COMMENT"]
    if 'INHERIT' in header and header['INHERIT']:
        h0 = fx[0].header
        for key in h0:
            if ( key not in blacklist ) and ( key not in header ):
                header[key] = h0[key]

    if "fill_header" in kwargs :
        hdus = kwargs["fill_header"]

        if hdus is None :
            hdus=[0,]
            if "PLC" in fx :
                hdus.append("PLC")
        
        if hdus is not None :
            log.info("will add header keywords from hdus %s"%str(hdus))
            for hdu in hdus :
                try :
                    ihdu = int(hdu)
                    hdu = ihdu
                except ValueError:
                    pass
                if hdu in fx :
                    hdu_header = fx[hdu].header
                    for key in hdu_header:
                        if ( key not in blacklist ) and ( key not in header ) :
                            log.debug("adding {} = {}".format(key,hdu_header[key]))
                            header[key] = hdu_header[key]                        
                        else :
                            log.debug("key %s already in header or blacklisted"%key)
                else :
                    log.warning("warning HDU %s not in fits file"%str(hdu))

        kwargs.pop("fill_header")
    
    fx.close()

    img = desispec.preproc.preproc(rawimage, header, primary_header, **kwargs)
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
        camera : b0, r1 .. z9 - override value in header
        primary_header : header to write in HDU0 if filename doesn't yet exist

    The primary utility of this function over raw fits calls is to ensure
    that all necessary keywords are present before writing the file.
    CCDSECx, BIASSECx, DATASECx where x=1,2,3, or 4
    DATE-OBS, GAINx and RDNOISEx will generate a non-fatal warning if missing
    '''
    log = get_logger()

    header = desispec.io.util.fitsheader(header)
    primary_header = desispec.io.util.fitsheader(primary_header)

    if rawdata.dtype not in (np.int16, np.int32, np.int64):
        message = 'dtype {} not supported for raw data'.format(rawdata.dtype)
        log.fatal(message)
        raise ValueError(message)

    fail_message = ''
    for required_key in ['DOSVER', 'FEEVER', 'DETECTOR']:
        if required_key not in primary_header:
            if required_key in header:
                primary_header[required_key] = header[required_key]
            else:
                fail_message = fail_message + \
                    'Keyword {} must be in header or primary_header\n'.format(required_key)
    if fail_message != '':
        raise ValueError(fail_message)

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
            log.warning('Gain keyword {} missing; using 1.0'.format(keyword))
            header[keyword] = 1.0

    #- Missing RDNOISEx is warning but not error
    for amp in ['1', '2', '3', '4']:
        keyword = 'RDNOISE'+amp
        if keyword not in header:
            log.warning('Readnoise keyword {} missing'.format(keyword))

    #- Stop if any keywords are missing
    if len(missing_keywords) > 0:
        raise KeyError('missing required keywords {}'.format(missing_keywords))

    #- Set EXTNAME=camera
    if camera is not None:
        header['CAMERA'] = camera.lower()
        extname = camera.upper()
    else:
        if header['CAMERA'] != header['CAMERA'].lower():
            log.warning('Converting CAMERA {} to lowercase'.format(header['CAMERA']))
            header['CAMERA'] = header['CAMERA'].lower()
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
        log.warning('Image compression not supported for 64-bit; writing uncompressed')
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
        add_dependencies(primary_header)
        hdus.append(fits.PrimaryHDU(None, header=primary_header))
        hdus.append(dataHDU)
        hdus.writeto(filename)
