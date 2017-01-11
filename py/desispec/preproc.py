'''
Preprocess raw DESI exposures
'''

import re
import numpy as np
import scipy.interpolate
import yaml
import os.path
from pkg_resources import resource_exists, resource_filename

from desispec.image import Image
from desispec import cosmics
from desispec.maskbits import ccdmask
from desispec.log import get_logger
log = get_logger()

def _parse_date_obs(value):
    '''
    converts DATE-OBS keywork to int
    with for instance DATE-OBS=2016-12-21T18:06:21.268371-05:00
    '''
    m = re.search('(\d+)-(\d+)-(\d+)T', value)
    Y,M,D=tuple(map(int, m.groups()))
    dateobs=int(Y*10000+M*100+D)
    return dateobs

def _parse_sec_keyword(value):
    '''
    parse keywords like BIASSECB='[7:56,51:4146]' into python slices

    python and FITS have almost opposite conventions,
      * FITS 1-indexed vs. python 0-indexed
      * FITS upperlimit-inclusive vs. python upperlimit-exclusive
      * FITS[x,y] vs. python[y,x]

    i.e. BIASSEC2='[7:56,51:4146]' -> (slice(50,4146), slice(6,56))
    '''
    m = re.search('\[(\d+):(\d+)\,(\d+):(\d+)\]', value)
    if m is None:
        m = re.search('\[(\d+):(\d+)\, (\d+):(\d+)\]', value)
        if m is None :
            raise ValueError('unable to parse {} as [a:b, c:d]'.format(value))
    
    xmin, xmax, ymin, ymax = tuple(map(int, m.groups()))

    return np.s_[ymin-1:ymax, xmin-1:xmax]

def _clipped_std_bias(nsigma):
    '''
    Returns the bias on the standard deviation of a sigma-clipped dataset

    Divide by the returned bias to get a corrected value::

        a = nsigma
        bias = sqrt((integrate x^2 exp(-x^2/2), x=-a..a) / (integrate exp(-x^2/2), x=-a..a))
             = sqrt(1 - 2a exp(-a^2/2) / (sqrt(2pi) erf(a/sqrt(2))))

    See http://www.wolframalpha.com/input/?i=(integrate+x%5E2+exp(-x%5E2%2F2),+x+%3D+-a+to+a)+%2F+(integrate+exp(-x%5E2%2F2),+x%3D-a+to+a)
    '''
    from scipy.special import erf
    a = float(nsigma)
    stdbias = np.sqrt(1 - 2*a*np.exp(-a**2/2.) / (np.sqrt(2*np.pi) * erf(a/np.sqrt(2))))
    return stdbias

def _overscan(pix, nsigma=5, niter=3):
    '''
    returns overscan, readnoise from overscan image pixels

    Args:
        pix (ndarray) : overscan pixels from CCD image

    Optional:
        nsigma (float) : number of standard deviations for sigma clipping
        niter (int) : number of iterative refits
    '''
    #- normalized median absolute deviation as robust version of RMS
    #- see https://en.wikipedia.org/wiki/Median_absolute_deviation
    overscan = np.median(pix)
    absdiff = np.abs(pix - overscan)
    readnoise = 1.4826*np.median(absdiff)

    #- input pixels are integers, so iteratively refit
    for i in range(niter):
        absdiff = np.abs(pix - overscan)
        good = absdiff < nsigma*readnoise
        overscan = np.mean(pix[good])
        readnoise = np.std(pix[good])

    #- correct for bias from sigma clipping
    readnoise /= _clipped_std_bias(nsigma)

    return overscan, readnoise


def _global_background(image,patch_width=200) :
    '''
    determine background using a 2D median with square patches of width = width
    that are interpolated
    (does not subtract the background)
    
    Args:
       image is expected to be already preprocessed
       ( image = ((rawimage-bias-overscan)*gain)/pixflat )
    Options:
       patch_width (integer) size in pixels of the median square patches
    Returns background image with same shape as input image
    '''
    bkg=np.zeros_like(image)
    bins0=np.linspace(0,image.shape[0],float(image.shape[0])/patch_width).astype(int)
    bins1=np.linspace(0,image.shape[1],float(image.shape[1])/patch_width).astype(int)
    bkg_grid=np.zeros((bins0.size-1,bins1.size-1))
    for j in xrange(bins1.size-1) :
        for i in xrange(bins0.size-1) :
            bkg_grid[i,j]=np.median(image[bins0[i]:bins0[i+1],bins1[j]:bins1[j+1]])
    
    nodes0=bins0[:-1]+(bins0[1]-bins0[0])/2.
    nodes1=bins1[:-1]+(bins1[1]-bins0[0])/2.
    spline=scipy.interpolate.RectBivariateSpline(nodes0,nodes1,bkg_grid,kx=2, ky=2, s=0)
    return spline(np.arange(0,image.shape[0]),np.arange(0,image.shape[1]))
    
    
def _background(image,header,patch_width=200,stitch_width=10,stitch=False) :
    '''
    determine background using a 2D median with square patches of width = width
    that are interpolated and optionnally try and match the level of amplifiers
    (does not subtract the background)
    
    Args:
       image is expected to be already preprocessed
       ( image = ((rawimage-bias-overscan)*gain)/pixflat )
       header is used to read CCDSEC
    Options:
       patch_width (integer) size in pixels of the median square patches
       stitch_width (integer) width in pixels of amplifier edges to match level of amplifiers 
       stitch : do match level of amplifiers 
    Returns background image with same shape as input image
    
    '''
    
    
    log.info("fit a smooth background over the whole image with median patches of size %dx%d"%(patch_width,patch_width))    
    bkg=_global_background(image,patch_width)
    
    if stitch :
        
        tmp_image=image-bkg
        
        log.info("stitch amps one with the other using median patches of size %dx%d"%(stitch_width,patch_width))
        tmp_bkg=np.zeros_like(image)
        
        
        for edge in [[1,2,1],[3,4,1],[1,3,0],[2,4,0]] :
            
            amp0=edge[0]
            amp1=edge[1]
            axis=edge[2]
                

            ii0=_parse_sec_keyword(header['CCDSEC%d'%amp0])
            ii1=_parse_sec_keyword(header['CCDSEC%d'%amp1])
            pos=ii0[axis].stop
            bins=np.linspace(ii0[axis-1].start,ii0[axis-1].stop,float(ii0[axis-1].stop-ii0[axis-1].start)/patch_width).astype(int)
            delta=np.zeros((bins.size-1))
            for i in xrange(bins.size-1) :
                if axis==0 :            
                    delta[i]=np.median(tmp_image[pos-stitch_width:pos,bins[i]:bins[i+1]])-np.median(tmp_image[pos:pos+stitch_width,bins[i]:bins[i+1]])
                else :
                    delta[i]=np.median(tmp_image[bins[i]:bins[i+1],pos-stitch_width:pos])-np.median(tmp_image[bins[i]:bins[i+1],pos:pos+stitch_width])
            nodes=bins[:-1]+(bins[1]-bins[0])/2.
            
            log.info("AMPS %d:%d mean diff=%f"%(amp0,amp1,np.mean(delta)))
                            
            delta=np.interp(np.arange(ii0[axis-1].stop-ii0[axis-1].start),nodes,delta)
            
            # smooth ramp along axis of scale patch_width 
            w=float(patch_width)
            x=np.arange(ii1[axis].stop-ii1[axis].start)                        
            #ramp=(x<w)*(x>w/2)*((x-w)/(w))**2+(x<=w/2)*(0.5-(x/(w))**2) # peaks at 0.5
            ramp=0.5*np.ones(x.shape)
                
            if axis==0 :
                tmp_bkg[ii1] -= np.outer(ramp,delta)
            else :
                tmp_bkg[ii1] -= np.outer(delta,ramp)

            x=np.arange(ii0[axis].stop-ii0[axis].start)  
            #ramp=(x<w)*(x>w/2)*((x-w)/(w))**2+(x<=w/2)*(0.5-(x/(w))**2) # peaks at 0.5
            ramp=0.5*np.ones(x.shape)

            if axis==0 :
                tmp_bkg[ii0] += np.outer(ramp[::-1],delta)
            else :
                tmp_bkg[ii0] += np.outer(delta,ramp[::-1])
                
        bkg += tmp_bkg
        tmp_image=image-bkg
        log.info("refit smooth background over the whole image with median patches of size %dx%d"%(patch_width,patch_width))
        bkg+=_global_background(tmp_image,patch_width)
        
    
    
    log.info("done")
    return bkg

def read_ccd_calibration(header, primary_header, filename) :

    if not os.path.isfile(filename) :
        log.error("Cannot find calibration data file '%s'"%filename)
        raise IOError("Cannot find calibration data file '%s'"%filename)
    
    stream = open(filename, 'r')
    data   = yaml.load(stream)
    stream.close()
    cameraid=header["CAMERA"].lower()
    if not cameraid in data :
        log.error("Cannot find data for camera %s in filename %s"%(cameraid,filename))
        raise KeyError("Cannot find  data for camera %s in filename %s"%(cameraid,filename))
    log.info("Found data for camera %s in filename %s"%(cameraid,filename))
    data=data[cameraid]
    log.info("Finding matching version ...")
    dateobs=_parse_date_obs(primary_header["DATE-OBS"])
    dosver=primary_header["DOSVER"].strip()
    feever=header["FEEVER"].strip()
    ccdname=header["CCDNAME"].strip()
    
    log.info("DATE-OBS=%d"%dateobs)
    found=False
    for version in data :
        log.info("Checking version %s"%version)
        datebegin=int(data[version]["DATE-OBS-BEGIN"])
        if dateobs < datebegin :
            log.info("Skip version %s with DATE-OBS-BEGIN=%d > DATE-OBS=%d"%(datebegin,dateobs))
            continue
        if "DATE-OBS-END" in data[version] and data[version]["DATE-OBS-END"].lower() != "none" :
            dateend=int(data[version]["DATE-OBS-END"])
            if dateobs >= dateend :
                log.info("Skip version %s with DATE-OBS-END=%d <= DATE-OBS=%d"%(datebegin,dateobs))
                continue
        if dosver != data[version]["DOSVER"].strip() :
            log.info("Skip version %s with DOSVER=%s != %s "%(data[version]["DOSVER"],dosver))
            continue
        if feever != data[version]["FEEVER"].strip() :
            log.info("Skip version %s with FEEVER=%s != %s"%(data[version]["FEEVER"],feever))
            continue
        if ccdname != data[version]["CCDNAME"].strip() :
            log.info("Skip version %s with CCDNAME=%s != %s"%(data[version]["CCDNAME"],ccdname))
            continue
        
        log.info("Found data version %s for camera %s in %s"%(version,cameraid,filename))
        found=True
        data=data[version]

    if not found :
        log.error("Didn't find matching calibration data in %s"%(filename))
        raise KeyError("Didn't find matching calibration data in %s"%(filename))
    return data


def preproc(rawimage, header, primary_header, bias=False, dark=False, pixflat=False, mask=False, bkgsub=False, nocosmic=False, cosmics_nsig=6, cosmics_cfudge=3., cosmics_c2fudge=0.8,ccd_calibration_filename=None):

    '''
    preprocess image using metadata in header

    image = ((rawimage-bias-overscan)*gain)/pixflat

    Args:
        rawimage : 2D numpy array directly from raw data file
        header : dict-like metadata, e.g. from FITS header, with keywords
            CAMERA, BIASSECx, DATASECx, CCDSECx
            where x = A, B, C, D for each of the 4 amplifiers
            (also supports old naming convention 1, 2, 3, 4).


    Optional bias, pixflat, and mask can each be:
        False: don't apply that step
        True: use default calibration data for that night
        ndarray: use that array
        filename (str or unicode): read HDU 0 and use that
        DATE-OBS is required in header if bias, pixflat, or mask=True


    

    Optional background subtraction with median filtering if bkgsub=True
    
    Optional disabling of cosmic ray rejection if nocosmic=True
    
    Optional tuning of cosmic ray rejection parameters:
        cosmics_nsig: number of sigma above background required
        cosmics_cfudge: number of sigma inconsistent with PSF required
        cosmics_c2fudge:  fudge factor applied to PSF

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

    calibration_data = None

    if ccd_calibration_filename is None :
        srch_file = "data/ccd/ccd_calibration.yaml"
        if not resource_exists('desispec', srch_file):
            log.error("Cannot find CCD calibration file {:s}".format(srch_file))               
        else :
            ccd_calibration_filename=resource_filename('desispec', srch_file)

    if ccd_calibration_filename is not None and  ccd_calibration_filename is not False :
        calibration_data = read_ccd_calibration(header, primary_header, ccd_calibration_filename)
    

    #- TODO: Check for required keywords first

    #- Subtract bias image
    camera = header['CAMERA'].lower()

    #- convert rawimage to float64 : this is the output format of read_image
    rawimage = rawimage.astype(np.float64)
    
    if bias is not False and bias is not None:
        if bias is True:
            #- use default bias file for this camera/night
            dateobs = header['DATE-OBS']
            bias = read_bias(camera=camera, dateobs=dateobs)
        ### elif isinstance(bias, (str, unicode)):
        elif isinstance(bias, str):
            #- treat as filename
            bias = read_bias(filename=bias)

        if bias.shape == rawimage.shape:
            rawimage = rawimage - bias
        else:
            raise ValueError('shape mismatch bias {} != rawimage {}'.format(bias.shape, rawimage.shape))


    if calibration_data and "AMPLIFIERS" in calibration_data :
        amp_ids=list(calibration_data["AMPLIFIERS"])
    else :
        amp_ids=['A','B','C','D']
    
    #- check whether it's indeed CCDSECx with x in ['A','B','C','D']
    #  or older version with x in ['1','2','3','4']
    #  we can remove this piece of code at later times 
    has_valid_keywords = True
    for amp in amp_ids :
        if not 'CCDSEC%s'%amp in header :
            log.warning("No CCDSEC%s keyword in header , will look for alternative naming CCDSEC{1,2,3,4} ..."%amp)
            has_valid_keywords = False
            break
    if not has_valid_keywords :
        amp_ids=['1','2','3','4']
        for amp in ['1','2','3','4'] :
            if not 'CCDSEC%s'%amp in header :
                log.error("No CCDSEC%s keyword, exit"%amp)
                raise KeyError("No CCDSEC%s keyword"%amp)
    
    #- Output arrays
    ny=0
    nx=0
    for amp in amp_ids :
        yy, xx = _parse_sec_keyword(header['CCDSEC%s'%amp])
        ny=max(ny,yy.stop)
        nx=max(nx,xx.stop)
    image = np.zeros( (ny,nx) )
    
    readnoise = np.zeros_like(image)
    
    #- Load mask
    if mask is not False and mask is not None:
        if mask is True:
            dateobs = header['DATE-OBS']
            mask = read_mask(camera=camera, dateobs=dateobs)
        ### elif isinstance(mask, (str, unicode)):
        elif isinstance(mask, str):
            mask = read_mask(filename=mask)
    else:
        mask = np.zeros(image.shape, dtype=np.int32)

    if mask.shape != image.shape:
        raise ValueError('shape mismatch mask {} != image {}'.format(mask.shape, image.shape))

    #- Load dark if exists
    if dark is not False and dark is not None:
        if dark is True:
            dateobs = header['DATE-OBS']
            dark = read_dark(camera=camera, dateobs=dateobs)
        elif isinstance(dark, str):
            #- treat as filename
            dark = read_dark(dark)
            if dark.shape != image.shape :
                log.error('shape mismatch dark {} != image {}'.format(dark.shape, image.shape))
                raise ValueError('shape mismatch dark {} != image {}'.format(dark.shape, image.shape))
        exptime =  primary_header['EXPTIME']
        log.info("Multiplying dark by exptime %f"%(exptime))
        dark *= exptime
                
            
    
    for amp in amp_ids :
        ii = _parse_sec_keyword(header['BIASSEC'+amp])

        #- Initial teststand data may be missing GAIN* keywords; don't crash
        if 'GAIN'+amp in header:
            gain = header['GAIN'+amp]          #- gain = electrons / ADU
        else:
            if calibration_data  and 'GAIN'+amp in calibration_data :
                gain = float(calibration_data['GAIN'+amp])
                log.info('Using GAIN{}={} from calibration data'.format(amp,gain))
            else :
                gain = 1.0
                log.warning('Missing keyword GAIN{} in header and nothing in calib data; using {}'.format(amp,gain))
                
        
        #- Add saturation level
        if 'SATURLEV'+amp in header:
            saturlev = header['SATURLEV'+amp]          # in electrons
        else:
            if calibration_data and 'SATURLEV'+amp in calibration_data :
                saturlev = float(calibration_data['SATURLEV'+amp])
                log.info('Using SATURLEV{}={} from calibration data'.format(amp,saturlev))
            else :
                saturlev = 200000
                log.warning('Missing keyword SATURLEV{} in header and nothing in calib data; using 200000'.format(amp,saturlev))
        

        overscan, rdnoise = _overscan(rawimage[ii])
        rdnoise *= gain
        kk = _parse_sec_keyword(header['CCDSEC'+amp])
        readnoise[kk] = rdnoise

        header['OVERSCN'+amp] = overscan
        header['OBSRDN'+amp] = rdnoise

        #- Warn/error if measured readnoise is very different from expected if exists
        if 'RDNOISE'+amp in header:
            expected_readnoise = header['RDNOISE'+amp]
            if rdnoise < 0.5*expected_readnoise:
                log.error('Amp {} measured readnoise {:.2f} < 0.5 * expected readnoise {:.2f}'.format(
                    amp, rdnoise, expected_readnoise))
            elif rdnoise < 0.9*expected_readnoise:
                log.warning('Amp {} measured readnoise {:.2f} < 0.9 * expected readnoise {:.2f}'.format(
                    amp, rdnoise, expected_readnoise))
            elif rdnoise > 2.0*expected_readnoise:
                log.error('Amp {} measured readnoise {:.2f} > 2 * expected readnoise {:.2f}'.format(
                    amp, rdnoise, expected_readnoise))
            elif rdnoise > 1.2*expected_readnoise:
                log.warning('Amp {} measured readnoise {:.2f} > 1.2 * expected readnoise {:.2f}'.format(
                    amp, rdnoise, expected_readnoise))
        #else:
        #    log.warning('Expected readnoise keyword {} missing'.format('RDNOISE'+amp))
        
        log.info("Measured readnoise for AMP %s = %f"%(amp,rdnoise))
        
        #- subtract overscan from data region and apply gain
        jj = _parse_sec_keyword(header['DATASEC'+amp])
        data = rawimage[jj] - overscan

        #- apply saturlev (defined in ADU), prior to multiplication by gain
        saturated = (rawimage[jj]>=saturlev)
        mask[kk][saturated] |= ccdmask.SATURATED
        
        #- subtract dark prior to multiplication by gain
        if dark is not False and dark is not None :
            data -= dark[kk]
        
        image[kk] = data*gain

        

    
    #- Divide by pixflat image
    if pixflat is not False and pixflat is not None:
        if pixflat is True:
            dateobs = header['DATE-OBS']
            pixflat = read_pixflat(camera=camera, dateobs=dateobs)
        ### elif isinstance(pixflat, (str, unicode)):
        elif isinstance(pixflat, str):
            pixflat = read_pixflat(filename=pixflat)

        if pixflat.shape != image.shape:
            raise ValueError('shape mismatch pixflat {} != image {}'.format(pixflat.shape, image.shape))

        if np.all(pixflat != 0.0):
            image /= pixflat
            readnoise /= pixflat
        else:
            good = (pixflat != 0.0)
            image[good] /= pixflat[good]
            readnoise[good] /= pixflat[good]
            mask[~good] |= ccdmask.PIXFLATZERO

        lowpixflat = (0 < pixflat) & (pixflat < 0.1)
        if np.any(lowpixflat):
            mask[lowpixflat] |= ccdmask.PIXFLATLOW

    #- Inverse variance, estimated directly from the data (BEWARE: biased!)
    var = image.clip(0) + readnoise**2
    ivar = 1.0 / var

    if bkgsub :
        bkg = _background(image,header)
        image -= bkg
        

    img = Image(image, ivar=ivar, mask=mask, meta=header, readnoise=readnoise, camera=camera)

    #- update img.mask to mask cosmic rays

    if not nocosmic :
        cosmics.reject_cosmic_rays(img,nsig=cosmics_nsig,cfudge=cosmics_cfudge,c2fudge=cosmics_c2fudge)
    
    return img

#-------------------------------------------------------------------------
#- The following I/O routines are here instead of desispec.io to avoid a
#- circular dependency between io and preproc:
#- io.read_raw -> preproc.preproc -> io.read_bias (bad)

def read_bias(filename=None, camera=None, dateobs=None):
    '''
    Return calibration bias filename for camera on dateobs or night

    Options:
        filename : input filename to read
        camera : e.g. 'b0', 'r1', 'z9'
        dateobs : DATE-OBS string, e.g. '2018-09-23T08:17:03.988'

    Notes:
        must provide filename, or both camera and dateobs
    '''
    from astropy.io import fits
    if filename is None:
        #- use camera and dateobs to derive what bias file should be used
        raise NotImplementedError
    else:
        return fits.getdata(filename, 0)

def read_pixflat(filename=None, camera=None, dateobs=None):
    '''
    Read calibration pixflat image for camera on dateobs.

    Options:
        filename : input filename to read
        camera : e.g. 'b0', 'r1', 'z9'
        dateobs : DATE-OBS string, e.g. '2018-09-23T08:17:03.988'

    Notes:
        must provide filename, or both camera and dateobs
    '''
    from astropy.io import fits
    if filename is None:
        #- use camera and dateobs to derive what pixflat file should be used
        raise NotImplementedError
    else:
        return fits.getdata(filename, 0)

def read_dark(filename=None, camera=None, dateobs=None):
    '''
    Read calibration dark image for camera on dateobs.

    Options:
        filename : input filename to read
        camera : e.g. 'b0', 'r1', 'z9'
        dateobs : DATE-OBS string, e.g. '2018-09-23T08:17:03.988'

    Notes:
        must provide filename, or both camera and dateobs
    '''
    from astropy.io import fits
    if filename is None:
        #- use camera and dateobs to derive what pixflat file should be used
        raise NotImplementedError
    else:
        return fits.getdata(filename, 0)

def read_mask(filename=None, camera=None, dateobs=None):
    '''
    Read bad pixel mask image for camera on dateobs.

    Options:
        filename : input filename to read
        camera : e.g. 'b0', 'r1', 'z9'
        dateobs : DATE-OBS string, e.g. '2018-09-23T08:17:03.988'

    Notes:
        must provide filename, or both camera and dateobs
    '''
    from astropy.io import fits
    if filename is None:
        #- use camera and dateobs to derive what mask file should be used
        raise NotImplementedError
    else:
        return fits.getdata(filename, 0)
