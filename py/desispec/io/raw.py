'''
I/O for DESI raw data files

See DESI-1229 for format details
TODO: move into datamodel after we have verified the format
'''

import os.path
from astropy.io import fits
import numpy as np

from desiutil.depend import add_dependencies

import desispec.io
import desispec.io.util
from desispec.util import header2night
import desispec.preproc
from desiutil.log import get_logger
from desispec.calibfinder import parse_date_obs, CalibFinder 
import desispec.maskbits as maskbits

def read_raw(filename, camera, fibermapfile=None, **kwargs):
    '''
    Returns preprocessed raw data from `camera` extension of `filename`

    Args:
        filename : input fits filename with DESI raw data
        camera : camera name (B0,R1, .. Z9) or FITS extension name or number

    Options:
        fibermapfile : read fibermap from this file; if None create blank fm
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
            if hdu > 0:
                log.warning("Did not find header keyword EXPTIME in hdu {}, moving to the next".format(hdu))
            hdu +=1 
        else :
            log.error("Did not find header keyword EXPTIME in any HDU of {}".format(filename))
            raise KeyError("Did not find header keyword EXPTIME in any HDU of {}".format(filename))

    #- Check if NIGHT keyword is present and valid; fix if needed
    try:
        tmp = int(primary_header['NIGHT'])
    except (KeyError, ValueError, TypeError):
        primary_header['NIGHT'] = header2night(primary_header)

    try:
        tmp = int(header['NIGHT'])
    except (KeyError, ValueError, TypeError):
        header['NIGHT'] = header2night(header)

    #- early data (e.g. 20200219/51053) had a mix of int vs. str NIGHT
    primary_header['NIGHT'] = int(primary_header['NIGHT'])
    header['NIGHT'] = int(header['NIGHT'])

    if primary_header['NIGHT'] != header['NIGHT']:
        msg = 'primary header NIGHT={} != camera header NIGHT={}'.format(
            primary_header['NIGHT'], header['NIGHT'])
        log.error(msg)
        raise ValueError(msg)

    #- early data have >8 char FIBERASSIGN key; rename to match current data
    if 'FIBERASSIGN' in primary_header:
        log.warning('renaming long header keyword FIBERASSIGN -> FIBASSGN')
        primary_header['FIBASSGN'] = primary_header['FIBERASSIGN']
        del primary_header['FIBERASSIGN']

    if 'FIBERASSIGN' in header:
        header['FIBASSGN'] = header['FIBERASSIGN']
        del header['FIBERASSIGN']

    skipkeys = ["EXTEND","SIMPLE","NAXIS1","NAXIS2","CHECKSUM","DATASUM","XTENSION","EXTNAME","COMMENT"]
    if 'INHERIT' in header and header['INHERIT']:
        h0 = fx[0].header
        for key in h0:
            if ( key not in skipkeys ) and ( key not in header ):
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
                        if ( key not in skipkeys ) and ( key not in header ) :
                            log.debug("adding {} = {}".format(key,hdu_header[key]))
                            header[key] = hdu_header[key]                        
                        else :
                            log.debug("key %s already in header or in skipkeys"%key)
                else :
                    log.warning("warning HDU %s not in fits file"%str(hdu))

        kwargs.pop("fill_header")

    fx.close()

    img = desispec.preproc.preproc(rawimage, header, primary_header, **kwargs)

    if fibermapfile is not None and os.path.exists(fibermapfile):
        fibermap = desispec.io.read_fibermap(fibermapfile)
    else:
        log.warning('creating blank fibermap')
        fibermap = desispec.io.empty_fibermap(5000)

    #- Add image header keywords inherited from raw data to fibermap too
    desispec.io.util.addkeys(fibermap.meta, img.meta)

    #- Augment the image header with some tile info from fibermap if needed
    for key in ['TILEID', 'TILERA', 'TILEDEC']:
        if key in fibermap.meta:
            if key not in img.meta:
                log.info('Updating header from fibermap {}={}'.format(
                    key, fibermap.meta[key]))
                img.meta[key] = fibermap.meta[key]
            elif img.meta[key] != fibermap.meta[key]:
                #- complain loudly, but don't crash and don't override
                log.error('Inconsistent {}: raw header {} != fibermap header {}'.format(key, img.meta[key], fibermap.meta[key]))


    #- Trim to matching camera based upon PETAL_LOC, but that requires
    #- a mapping prior to 20191211

    #- HACK HACK HACK
    #- TODO: replace this with a mapping from calibfinder, as soon as
    #- that is implemented in calibfinder / desi_spectro_calib
    #- HACK HACK HACK

    #- From DESI-5286v5 page 3 where sp=sm-1 and
    #- "spectro logical number" = petal_loc
    spec_to_petal = {4:2, 2:9, 3:0, 5:3, 1:8, 0:4, 6:6, 7:7, 8:5, 9:1}
    assert set(spec_to_petal.keys()) == set(range(10))
    assert set(spec_to_petal.values()) == set(range(10))

    #- Mapping only for dates < 20191211
    if "NIGHT" in primary_header:
        dateobs = int(primary_header["NIGHT"])
    elif "DATE-OBS" in primary_header:
        dateobs=parse_date_obs(primary_header["DATE-OBS"])
    else:
        msg = "Need either NIGHT or DATE-OBS in primary header"
        log.error(msg)
        raise KeyError(msg)
    if dateobs < 20191211 :
        petal_loc = spec_to_petal[int(camera[1])]
        log.warning('Prior to 20191211, mapping camera {} to PETAL_LOC={}'.format(camera, petal_loc))
    else :
        petal_loc = int(camera[1])
        log.debug('Since 20191211, camera {} is PETAL_LOC={}'.format(camera, petal_loc))
    
    ii = (fibermap['PETAL_LOC'] == petal_loc)
    fibermap = fibermap[ii]

    ## Mask fibers
    cfinder = CalibFinder([header,primary_header])
    mod_fibers = fibermap['FIBER'].data % 500

    ## Mask blacklisted fibers
    fiberblacklist = cfinder.fiberblacklist()
    for fiber in fiberblacklist:
        loc = np.where(mod_fibers==fiber)[0]
        fibermap['FIBERSTATUS'][loc] |= maskbits.fibermask.BADFIBER

    # Mask Fibers that are set to be excluded due to CCD/amp/readout issues
    camname = camera.upper()[0]
    if camname == 'B':
        badamp_bit = maskbits.fibermask.BADAMPB
    elif camname == 'R':
        badamp_bit = maskbits.fibermask.BADAMPR
    else:
        #elif camname == 'Z':
        badamp_bit = maskbits.fibermask.BADAMPZ

    fibers_to_exclude = cfinder.fibers_to_exclude()
    for fiber in fibers_to_exclude:
        loc = np.where(mod_fibers==fiber)[0]
        fibermap['FIBERSTATUS'][loc] |= badamp_bit        

    img.fibermap = fibermap

    return img

def write_raw(filename, rawdata, header, camera=None, primary_header=None):
    '''
    Write raw pixel data to a DESI raw data file

    Args:
        filename : file name to write data; if this exists, append a new HDU
        rawdata : 2D ndarray of raw pixel data including overscans
        header : dict-like object or fits.Header with keywords
            CCDSECx, BIASSECx, DATASECx where x=A,B,C,D

    Options:
        camera : b0, r1 .. z9 - override value in header
        primary_header : header to write in HDU0 if filename doesn't yet exist

    The primary utility of this function over raw fits calls is to ensure
    that all necessary keywords are present before writing the file.
    CCDSECx, BIASSECx, DATASECx where x=A,B,C,D
    DATE-OBS will generate a non-fatal warning if missing
    '''
    log = get_logger()

    header = desispec.io.util.fitsheader(header)
    primary_header = desispec.io.util.fitsheader(primary_header)

    header['FILENAME'] = filename

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

    ampnames = ['A', 'B', 'C', 'D']  #- previously 1,2,3,4
    for amp in ampnames:
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
