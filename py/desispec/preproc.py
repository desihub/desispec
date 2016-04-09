'''
Preprocess raw DESI exposures
'''

import re
import numpy as np
from desispec.image import Image

from desispec import cosmics
import desispec.io
from desispec.log import get_logger
log = get_logger()

def _parse_sec_keyword(value):
    '''
    parse keywords like BIASSECB='[7:56,51:4146]' into python slices
    
    python and FITS have almost opposite conventions,
      * FITS 1-indexed vs. python 0-indexed
      * FITS upperlimit-inclusive vs. python upperlimit-exclusive
      * FITS[x,y] vs. python[y,x]
    
    i.e. BIASSECB='[7:56,51:4146]' -> (slice(50,4146), slice(6,56))
    '''
    m = re.search('\[(\d+):(\d+)\,(\d+):(\d+)\]', value)
    if m is None:
        raise ValueError, 'unable to parse {} as [a:b, c:d]'.format(value)
    else:
        xmin, xmax, ymin, ymax = map(int, m.groups())
        
    return np.s_[ymin-1:ymax, xmin-1:xmax]

def _overscan(pix, sigma=5, niter=3):
    '''
    returns overscan, readnoise from overscan image pixels
    
    Optional:
        sigma : sigma clipping parameter for mean calculation
        niter : number of iterative refits
    '''
    #- normalized median absolute deviation as robust version of RMS
    #- see https://en.wikipedia.org/wiki/Median_absolute_deviation
    overscan = np.median(pix)
    absdiff = np.abs(pix - overscan)
    readnoise = 1.4826*np.median(absdiff)
    
    #- input pixels are integers, so iteratively refit
    for i in range(niter):
        absdiff = np.abs(pix - overscan)
        good = absdiff < sigma*readnoise
        overscan = np.mean(pix[good])
        readnoise = np.std(pix[good])
    
    return overscan, readnoise

def preproc(rawimage, header, bias=False, pixflat=False, mask=False):
    '''
    preprocess image using metadata in header

    image = ((rawimage-bias-overscan)*gain)/pixflat
    
    Args:
        rawimage : 2D numpy array directly from raw data file
        header : dict-like metadata, e.g. from FITS header, with keywords
            CAMERA, DATE-OBS, CCDSEC, BIASSECx, DATASECx, CCDSECx
            where x = A, B, C, D for each of the 4 amplifiers
        
    Optional bias, pixflat, and mask can each be:
        False: don't apply that step
        True: use default calibration data for that night
        ndarray: use that array
        filename (str or unicode): read HDU 0 and use that

    Returns Image object with member variables:
        image : 2D preprocessed image in units of electrons per pixel
        ivar : 2D inverse variance of image
        mask : 2D mask of image (0=good)
        readnoise : 2D per-pixel readnoise of image
        meta : metadata dictionary
            TODO: define what keywords are included

    preprocessing includes the following steps:
        - bias image subtraction
        - overscan subtraction (from BIASSEC* keyword defined regions)
        - readnoise estimation (from BIASSEC* keyword defined regions)
        - gain correction (from GAIN* keywords)
        - pixel flat correction
        - cosmic ray masking
        - propagation of input known bad pixel mask
        - inverse variance estimation
        
    Notes:
    
    The bias image is subtracted before any other calculation to remove any
    non-uniformities in the overscan regions prior to calculating overscan
    levels and readnoise.
    
    The readnoise is an image not just one number per amp, because the pixflat
    image also affects the interpreted readnoise.
    
    The inverse variance is estimated from the readnoise and the image itself,
    and thus is biased.
    '''
    #- TODO: Check for required keywords first

    #- Subtract bias image
    camera = header['CAMERA']
    dateobs = header['DATE-OBS']
    
    if bias is not False and bias is not None:
        if bias is True:
            #- use default bias file for this camera/night
            bias = desispec.io.preproc.read_bias(camera=camera, dateobs=dateobs)
        elif isinstance(bias, (str, unicode)):
            #- treat as filename
            bias = desispec.io.preproc.read_bias(filename=bias)
            
        if bias.shape == rawimage.shape:
            rawimage -= bias
        else:
            raise ValueError('shape mismatch bias {} != rawimage {}'.format(bias.shape, rawimage.shape))

    #- Output arrays
    yy, xx = _parse_sec_keyword(header['CCDSEC'])
    image = np.zeros( (yy.stop, xx.stop) )
    readnoise = np.zeros_like(image)
    
    for amp in ['A', 'B', 'C', 'D']:
        ii = _parse_sec_keyword(header['BIASSEC'+amp])
        gain = header['GAIN'+amp]          #- gain = electrons / ADU

        overscan, rdnoise = _overscan(rawimage[ii])
        rdnoise *= gain
        kk = _parse_sec_keyword(header['CCDSEC'+amp])
        readnoise[kk] = rdnoise
        
        header['OVERSCN'+amp] = overscan
        header['OBSRDN'+amp] = rdnoise
        
        #- Warn/error if measured readnoise is very different from expected
        if 'RDNOISE'+amp in header:
            expected_readnoise = header['RDNOISE'+amp]
            if rdnoise < 0.5*expected_readnoise:
                log.error('Amp {} measured readnoise {:.2f} < 0.5 * expected readnoise {:.2f}'.format(
                    amp, rdnoise, expected_readnoise))
            elif rdnoise < 0.9*expected_readnoise:
                log.warn('Amp {} measured readnoise {:.2f} < 0.9 * expected readnoise {:.2f}'.format(
                    amp, rdnoise, expected_readnoise))
            elif rdnoise > 2.0*expected_readnoise:
                log.error('Amp {} measured readnoise {:.2f} > 2 * expected readnoise {:.2f}'.format(
                    amp, rdnoise, expected_readnoise))
            elif rdnoise > 1.2*expected_readnoise:
                log.warn('Amp {} measured readnoise {:.2f} > 1.2 * expected readnoise {:.2f}'.format(
                    amp, rdnoise, expected_readnoise))
        else:
            log.warn('Expected readnoise keyword {} missing'.format('RDNOISE'+amp))
        
        #- subtract overscan from data region and apply gain
        jj = _parse_sec_keyword(header['DATASEC'+amp])
        data = rawimage[jj] - overscan
        image[kk] = data*gain

    #- Divide by pixflat image
    if pixflat is not False and pixflat is not None:
        if pixflat is True:
            pixflat = desispec.io.preproc.read_pixflat(camera=camera, dateobs=dateobs)
        elif isinstance(pixflat, (str, unicode)):
            pixflat = desispec.io.preproc.read_pixflat(filename=pixflat)

        if pixflat.shape != image.shape:
            raise ValueError('shape mismatch pixflat {} != image {}'.format(pixflat.shape, image.shape))

        image /= pixflat
        readnoise /= pixflat

    #- Load mask
    if mask is not False and mask is not None:
        if mask is True:
            mask = desispec.io.preproc.read_mask(camera=camera, dateobs=dateobs)
        elif isinstance(mask, (str, unicode)):
            mask = desispec.io.preproc.read_mask(filename=mask)
    else:
        mask = np.zeros(image.shape, dtype=np.int32)
        
    if mask.shape != image.shape:
        raise ValueError('shape mismatch mask {} != image {}'.format(mask.shape, image.shape))

    #- Inverse variance, estimated directly from the data (BEWARE: biased!)
    var = image.clip(0) + readnoise**2
    ivar = 1.0 / var
    
    img = Image(image, ivar=ivar, mask=mask, meta=header, readnoise=readnoise, camera=camera)

    #- update img.mask to mask cosmic rays
    cosmics.reject_cosmic_rays(img)
    
    return img