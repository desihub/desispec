'''
Preprocess raw DESI exposures
'''

import re
import os
import numpy as np
import scipy.interpolate
from pkg_resources import resource_exists, resource_filename

from scipy import signal

from desispec.image import Image
from desispec import cosmics
from desispec.maskbits import ccdmask
from desiutil.log import get_logger
from desiutil import depend
from desispec.calibfinder import CalibFinder
from desispec.darktrail import correct_dark_trail
from desispec.scatteredlight import model_scattered_light
from desispec.io.xytraceset import read_xytraceset
from desispec.io import read_fiberflat, shorten_filename
from desispec.io.util import addkeys
from desispec.maskedmedian import masked_median
from desispec.image_model import compute_image_model

# log = get_logger()

def get_amp_ids(header):
    '''
    Return list of amp names ['A','B','C','D'] or ['1','2','3','4']
    based upon header keywords
    '''
    if 'CCDSECA' in header:
        amp_ids = ['A', 'B', 'C', 'D']
    elif 'CCDSEC1' in header:
        amp_ids = ['1', '2', '3', '4']
    else:
        log = get_logger()
        message = "No CCDSECA or CCDSEC1; Can't derive amp names from header"
        log.fatal(message)
        raise KeyError(message)

    return amp_ids


def _parse_sec_keyword(value):
    log = get_logger()
    log.warning('please use parse_sec_keyword (no underscore)')
    return parse_sec_keyword(value)

def parse_sec_keyword(value):
    '''
    parse keywords like BIASSECB='[7:56,51:4146]' into python slices

    python and FITS have almost opposite conventions,
      * FITS 1-indexed vs. python 0-indexed
      * FITS upperlimit-inclusive vs. python upperlimit-exclusive
      * FITS[x,y] vs. python[y,x]

    i.e. BIASSEC2='[7:56,51:4146]' -> (slice(50,4146), slice(6,56))
    '''
    m = re.search(r'\[(\d+):(\d+)\,(\d+):(\d+)\]', value)
    if m is None:
        m = re.search(r'\[(\d+):(\d+)\, (\d+):(\d+)\]', value)
        if m is None :
            raise ValueError('unable to parse {} as [a:b, c:d]'.format(value))

    xmin, xmax, ymin, ymax = tuple(map(int, m.groups()))

    return np.s_[ymin-1:ymax, xmin-1:xmax]

def hello() :
    print("hello")

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
    """
    Calculates overscan, readnoise from overscan image pixels

    Args:
        pix (ndarray) : overscan pixels from CCD image

    Optional:
        nsigma (float) : number of standard deviations for sigma clipping
        niter (int) : number of iterative refits

    Returns:
        overscan (float): Mean, sigma-clipped value
        readnoise (float):

    """
    log=get_logger()
    #- normalized median absolute deviation as robust version of RMS
    #- see https://en.wikipedia.org/wiki/Median_absolute_deviation
    overscan = np.median(pix)
    absdiff = np.abs(pix - overscan)
    readnoise = 1.4826*np.median(absdiff)

    #- input pixels are integers, so iteratively refit
    for i in range(niter):
        absdiff = np.abs(pix - overscan)
        good = absdiff < nsigma*readnoise
        if np.sum(good)<5 :
            log.error("error in sigma clipping for overscan measurement, return result without clipping")
            overscan = np.median(pix)
            absdiff = np.abs(pix - overscan)
            readnoise = 1.4826*np.median(absdiff)
            return overscan,readnoise
        overscan = np.mean(pix[good])
        readnoise = np.std(pix[good])

    #- correct for bias from sigma clipping
    readnoise /= _clipped_std_bias(nsigma)

    return overscan, readnoise


def _savgol_clipped(data, window=15, polyorder=5, niter=0, threshold=3.):
    """
    Simple method to iteratively do a SavGol filter
    with rejection and replacing rejected pixels by
    nearest neighbors

    Args:
        data (ndarray):
        window (int):  Window parameter for savgol
        polyorder (int):
        niter (int):
        threshold (float):

    Returns:

    """
    print("Window: {}".format(window))

    ### 1st estimation
    array = data.copy()
    fitted = signal.savgol_filter(array, window, polyorder)
    filtered = array - fitted
    ### nth iteration
    nrej = 0
    for i in range(niter):
        sigma = filtered.std(axis=0)
        mask = np.abs(filtered) >= threshold*sigma
        good = np.where(~mask)[0]
        # Replace with nearest neighbors
        new_nrej = np.sum(mask)
        if new_nrej == nrej:
            break
        else:
            nrej = new_nrej
        for imask in np.where(mask)[0]:
            # Replace with nearest neighbors
            i0 = np.max(good[good < imask])
            i1 = np.min(good[good > imask])
            array[imask] = np.mean([array[i0], array[i1]])
        ### Refit
        fitted = signal.savgol_filter(array, window, polyorder)
    # Return
    return fitted


def _global_background(image,patch_width=200) :
    '''
    determine background using a 2D median with square patches of width = width
    that are interpolated
    (does not subtract the background)

    Args:
       image (ndarray) is expected to be already preprocessed
       ( image = ((rawimage-bias-overscan)*gain)/pixflat )
    Options:
       patch_width (integer) size in pixels of the median square patches

    Returns background image with same shape as input image
    '''
    bkg=np.zeros_like(image)
    bins0=np.linspace(0,image.shape[0],float(image.shape[0])/patch_width).astype(int)
    bins1=np.linspace(0,image.shape[1],float(image.shape[1])/patch_width).astype(int)
    bkg_grid=np.zeros((bins0.size-1,bins1.size-1))
    for j in range(bins1.size-1) :
        for i in range(bins0.size-1) :
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
       image (ndarray) is expected to be already preprocessed
       ( image = ((rawimage-bias-overscan)*gain)/pixflat )
       header is used to read CCDSEC
    Options:
       patch_width (integer) size in pixels of the median square patches
       stitch_width (integer) width in pixels of amplifier edges to match level of amplifiers
       stitch : do match level of amplifiers

    Returns background image with same shape as input image
    '''
    log=get_logger()


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


            ii0=parse_sec_keyword(header['CCDSEC%d'%amp0])
            ii1=parse_sec_keyword(header['CCDSEC%d'%amp1])
            pos=ii0[axis].stop
            bins=np.linspace(ii0[axis-1].start,ii0[axis-1].stop,float(ii0[axis-1].stop-ii0[axis-1].start)/patch_width).astype(int)
            delta=np.zeros((bins.size-1))
            for i in range(bins.size-1) :
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

def get_calibration_image(cfinder, keyword, entry, header=None):
    """Reads a calibration file

    Args:
        cfinder : None or CalibFinder object
        keyword :  BIAS, MASK, or PIXFLAT
        entry : boolean or filename or image
                if entry==False return False
                if entry==True use calibration filename from calib. config and read it
                if entry==str use this for the filename
                if entry==image return input

    Options:
        header : if not None, update header['CAL...'] = calib provenance

    returns:
       2D numpy array with calibration image
    """
    log=get_logger()

    #- set the header to something so that we don't have to keep checking it
    if header is None:
        header = dict()

    calkey = 'CCD_CALIB_{}'.format(keyword.upper())
    if entry is False:
        depend.setdep(header, calkey, 'None')
        return False # we don't want do anything

    filename = None
    if entry is True :
        # we have to find the filename
        if cfinder is None :
            log.error("no calibration data was found")
            raise ValueError("no calibration data was found")
        if cfinder.haskey(keyword) :
            filename = cfinder.findfile(keyword)
            depend.setdep(header, calkey, shorten_filename(filename))
        else :
            depend.setdep(header, calkey, 'None')
            return False # we say in the calibration data we don't need this
    elif isinstance(entry,str) :
        filename = entry
        depend.setdep(header, calkey, shorten_filename(filename))
    else :
        depend.setdep(header, calkey, 'Unknown image')
        return entry # it's expected to be an image array

    log.info("Using %s %s"%(keyword,filename))
    if keyword == "BIAS" :
        return read_bias(filename=filename)
    elif keyword == "MASK" :
        return read_mask(filename=filename)
    elif keyword == "PIXFLAT" :
        return read_pixflat(filename=filename)
    elif keyword == "DARK" :
        raise ValueError("Dark are now treated separately.")
    else :
        log.error("Don't known how to read %s in %s"%(keyword,path))
        raise ValueError("Don't known how to read %s in %s"%(keyword,path))
    return False

def preproc(rawimage, header, primary_header, bias=True, dark=True, pixflat=True, mask=True,
            bkgsub=False, nocosmic=False, cosmics_nsig=6, cosmics_cfudge=3., cosmics_c2fudge=0.5,
            ccd_calibration_filename=None, nocrosstalk=False, nogain=False,
            overscan_per_row=False, use_overscan_row=False, use_savgol=None,
            nodarktrail=False,remove_scattered_light=False,psf_filename=None,
            bias_img=None,model_variance=False):

    '''
    preprocess image using metadata in header

    image = ((rawimage-bias-overscan)*gain)/pixflat

    Args:
        rawimage : 2D numpy array directly from raw data file
        header : dict-like metadata, e.g. from FITS header, with keywords
            CAMERA, BIASSECx, DATASECx, CCDSECx
            where x = A, B, C, D for each of the 4 amplifiers
            (also supports old naming convention 1, 2, 3, 4).
        primary_header: dict-like metadata fit keywords EXPTIME, DOSVER
            DATE-OBS is also required if bias, pixflat, or mask=True

    Optional bias, pixflat, and mask can each be:
        False: don't apply that step
        True: use default calibration data for that night
        ndarray: use that array
        filename (str or unicode): read HDU 0 and use that

    Optional overscan features:
        overscan_per_row : bool,  Subtract the overscan_col values
            row by row from the data.
        use_overscan_row : bool,  Subtract off the overscan_row
            from the data (default: False).  Requires ORSEC in
            the Header
        use_savgol : bool,  Specify whether to use Savitsky-Golay filter for
            the overscan.   (default: False).  Requires use_overscan_row=True
            to have any effect.

    Optional variance model if model_variance=True
    Optional background subtraction with median filtering if bkgsub=True

    Optional disabling of cosmic ray rejection if nocosmic=True
    Optional disabling of dark trail correction if nodarktrail=True

    Optional bias image (testing only) may be provided by bias_img=

    Optional tuning of cosmic ray rejection parameters:
        cosmics_nsig: number of sigma above background required
        cosmics_cfudge: number of sigma inconsistent with PSF required
        cosmics_c2fudge:  fudge factor applied to PSF

    Optional fit and subtraction of scattered light

    Returns Image object with member variables:
        pix : 2D preprocessed image in units of electrons per pixel
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
    log=get_logger()

    header = header.copy()
    depend.setdep(header, 'DESI_SPECTRO_CALIB', os.getenv('DESI_SPECTRO_CALIB'))

    cfinder = None

    if ccd_calibration_filename is not False:
        cfinder = CalibFinder([header, primary_header], yaml_file=ccd_calibration_filename)

    #- TODO: Check for required keywords first

    #- Subtract bias image
    camera = header['CAMERA'].lower()

    #- convert rawimage to float64 : this is the output format of read_image
    rawimage = rawimage.astype(np.float64)

    # Savgol
    if cfinder and cfinder.haskey("USE_ORSEC"):
        use_overscan_row = cfinder.value("USE_ORSEC")
    if cfinder and cfinder.haskey("SAVGOL"):
        use_savgol = cfinder.value("SAVGOL")

    # Set bias image, as desired
    if bias_img is None:
        bias = get_calibration_image(cfinder,"BIAS",bias,header)
    else:
        bias = bias_img

    #- Check if this file uses amp names 1,2,3,4 (old) or A,B,C,D (new)
    amp_ids = get_amp_ids(header)
    #- Double check that we have the necessary keywords
    missing_keywords = list()
    for prefix in ['CCDSEC', 'BIASSEC']:
        for amp in amp_ids :
            key = prefix+amp
            if not key in header :
                log.error('No {} keyword in header'.format(key))
                missing_keywords.append(key)

    if len(missing_keywords) > 0:
        raise KeyError("Missing keywords {}".format(' '.join(missing_keywords)))


    #- Output arrays
    ny=0
    nx=0
    for amp in amp_ids :
        yy, xx = parse_sec_keyword(header['CCDSEC%s'%amp])
        ny=max(ny,yy.stop)
        nx=max(nx,xx.stop)
    image = np.zeros((ny,nx))

    readnoise = np.zeros_like(image)

    #- Load dark
    if cfinder and cfinder.haskey("DARK") and (dark is not False):

        #- Exposure time
        if cfinder and cfinder.haskey("EXPTIMEKEY") :
            exptime_key=cfinder.value("EXPTIMEKEY")
            log.info("Using exposure time keyword %s for dark normalization"%exptime_key)
        else :
            exptime_key="EXPTIME"
        exptime =  primary_header[exptime_key]
        log.info("Use exptime = {} sec to compute the dark current".format(exptime))

        dark_filename = cfinder.findfile("DARK")
        depend.setdep(header, 'CCD_CALIB_DARK', shorten_filename(dark_filename))
        log.info(f'Using DARK model from {dark_filename}')
        # dark is multipled by exptime, or we use the non-linear dark model in the routine
        dark = read_dark(filename=dark_filename,exptime=exptime)

        if dark.shape == image.shape :
            log.info("dark is trimmed")
            trimmed_dark_in_electrons = dark
            dark_is_trimmed   = True
        elif dark.shape == rawimage.shape :
            log.info("dark is not trimmed")
            trimmed_dark_in_electrons = np.zeros_like(image)
            dark_is_trimmed = False
        else :
            message="incompatible dark shape={} when raw shape={} and preproc shape={}".format(dark.shape,rawimage.shape,image.shape)
            log.error(message)
            raise ValueError(message)

    else:
        dark = False

    if bias is not False : #- it's an array
        if bias.shape == rawimage.shape  :
            log.info("subtracting bias")
            rawimage = rawimage - bias
        else:
            raise ValueError('shape mismatch bias {} != rawimage {}'.format(bias.shape, rawimage.shape))

    #- Load mask
    mask = get_calibration_image(cfinder,"MASK",mask,header)

    if mask is False :
        mask = np.zeros(image.shape, dtype=np.int32)
    else :
        if mask.shape != image.shape :
            raise ValueError('shape mismatch mask {} != image {}'.format(mask.shape, image.shape))

    for amp in amp_ids:
        # Grab the sections
        ov_col = parse_sec_keyword(header['BIASSEC'+amp])
        if 'ORSEC'+amp in header.keys():
            ov_row = parse_sec_keyword(header['ORSEC'+amp])
        elif use_overscan_row:
            log.error('No ORSEC{} keyword; not using overscan_row'.format(amp))
            use_overscan_row = False

        if nogain :
            gain = 1.
        else :
            #- Initial teststand data may be missing GAIN* keywords; don't crash
            if 'GAIN'+amp in header:
                gain = header['GAIN'+amp]          #- gain = electrons / ADU
            else:
                if cfinder and cfinder.haskey('GAIN'+amp) :
                    gain = float(cfinder.value('GAIN'+amp))
                    log.info('Using GAIN{}={} from calibration data'.format(amp,gain))
                else :
                    gain = 1.0
                    log.warning('Missing keyword GAIN{} in header and nothing in calib data; using {}'.format(amp,gain))

        #- Record what gain value was actually used
        header['GAIN'+amp] = gain

        #- Add saturation level
        if 'SATURLEV'+amp in header:
            saturlev_adu = header['SATURLEV'+amp]          # in ADU
        else:
            if cfinder and cfinder.haskey('SATURLEV'+amp) :
                saturlev_adu = float(cfinder.value('SATURLEV'+amp))
                log.info('Using SATURLEV{}={} from calibration data'.format(amp,saturlev_adu))
            else :
                saturlev_adu = 2**16-1 # 65535 is the max value in the images
                log.warning('Missing keyword SATURLEV{} in header and nothing in calib data; using {} ADU'.format(amp,saturlev_adu))
        header['SATULEV'+amp] = (saturlev_adu,"saturation or non lin. level, in ADU, inc. bias")


        # Generate the overscan images
        raw_overscan_col = rawimage[ov_col].copy()

        if use_overscan_row:
            raw_overscan_row = rawimage[ov_row].copy()
            overscan_row = np.zeros_like(raw_overscan_row)

            # Remove overscan_col from overscan_row
            raw_overscan_squared = rawimage[ov_row[0], ov_col[1]].copy()
            for row in range(raw_overscan_row.shape[0]):
                o,r = _overscan(raw_overscan_squared[row])
                overscan_row[row] = raw_overscan_row[row] - o

        # Now remove the overscan_col
        nrows=raw_overscan_col.shape[0]
        log.info("nrows in overscan=%d"%nrows)
        overscan_col = np.zeros(nrows)
        rdnoise  = np.zeros(nrows)
        if (cfinder and cfinder.haskey('OVERSCAN'+amp) and cfinder.value("OVERSCAN"+amp).upper()=="PER_ROW") or overscan_per_row:
            log.info("Subtracting overscan per row for amplifier %s of camera %s"%(amp,camera))
            for j in range(nrows) :
                if np.isnan(np.sum(overscan_col[j])) :
                    log.warning("NaN values in row %d of overscan of amplifier %s of camera %s"%(j,amp,camera))
                    continue
                o,r =  _overscan(raw_overscan_col[j])
                #log.info("%d %f %f"%(j,o,r))
                overscan_col[j]=o
                rdnoise[j]=r
        else :
            log.info("Subtracting average overscan for amplifier %s of camera %s"%(amp,camera))
            o,r =  _overscan(raw_overscan_col)
            overscan_col += o
            rdnoise  += r

        rdnoise *= gain
        median_rdnoise  = np.median(rdnoise)
        median_overscan = np.median(overscan_col)
        log.info("Median rdnoise and overscan= %f %f"%(median_rdnoise,median_overscan))

        kk = parse_sec_keyword(header['CCDSEC'+amp])
        for j in range(nrows) :
            readnoise[kk][j] = rdnoise[j]

        header['OVERSCN'+amp] = (median_overscan,'ADUs (gain not applied)')
        if gain != 1 :
            rdnoise_message = 'electrons (gain is applied)'
            gain_message    = 'e/ADU (gain applied to image)'
        else :
            rdnoise_message = 'ADUs (gain not applied)'
            gain_message    = 'gain not applied to image'
        header['OBSRDN'+amp] = (median_rdnoise,rdnoise_message)
        header['GAIN'+amp] = (gain,gain_message)

        #- Warn/error if measured readnoise is very different from expected if exists
        if 'RDNOISE'+amp in header:
            expected_readnoise = header['RDNOISE'+amp]
            if median_rdnoise < 0.5*expected_readnoise:
                log.error('Amp {} measured readnoise {:.2f} < 0.5 * expected readnoise {:.2f}'.format(
                    amp, median_rdnoise, expected_readnoise))
            elif median_rdnoise < 0.9*expected_readnoise:
                log.warning('Amp {} measured readnoise {:.2f} < 0.9 * expected readnoise {:.2f}'.format(
                    amp, median_rdnoise, expected_readnoise))
            elif median_rdnoise > 2.0*expected_readnoise:
                log.error('Amp {} measured readnoise {:.2f} > 2 * expected readnoise {:.2f}'.format(
                    amp, median_rdnoise, expected_readnoise))
            elif median_rdnoise > 1.2*expected_readnoise:
                log.warning('Amp {} measured readnoise {:.2f} > 1.2 * expected readnoise {:.2f}'.format(
                    amp, median_rdnoise, expected_readnoise))
        #else:
        #    log.warning('Expected readnoise keyword {} missing'.format('RDNOISE'+amp))

        log.info("Measured readnoise for AMP %s = %f"%(amp,median_rdnoise))

        #- subtract overscan from data region and apply gain
        jj = parse_sec_keyword(header['DATASEC'+amp])

        data = rawimage[jj].copy()
        # Subtract columns
        for k in range(nrows):
            data[k] -= overscan_col[k]

        saturlev_elec = gain*(saturlev_adu - np.mean(overscan_col))
        header['SATUELE'+amp] = (saturlev_elec,"saturation or non lin. level, in electrons")

        # And now the rows
        if use_overscan_row:
            # Savgol?
            if use_savgol:
                log.info("Using savgol")
                collapse_oscan_row = np.zeros(overscan_row.shape[1])
                for col in range(overscan_row.shape[1]):
                    o, _ = _overscan(overscan_row[:,col])
                    collapse_oscan_row[col] = o
                oscan_row = _savgol_clipped(collapse_oscan_row, niter=0)
                oimg_row = np.outer(np.ones(data.shape[0]), oscan_row)
                data -= oimg_row
            else:
                o,r = _overscan(overscan_row)
                data -= o

        #- apply saturlev (defined in ADU), prior to multiplication by gain
        saturated = (rawimage[jj]>=saturlev_adu)
        mask[kk][saturated] |= ccdmask.SATURATED

        #- ADC to electrons
        image[kk] = data*gain

        if dark is not False :
            if not dark_is_trimmed :
                trimmed_dark_in_electrons[kk] = dark[jj]*gain

    if not nocrosstalk :
        #- apply cross-talk

        # the ccd looks like :
        # C D
        # A B
        # for cross talk, we need a symmetric 4x4 flip_matrix
        # of coordinates ABCD giving flip of both axis
        # when computing crosstalk of
        #    A   B   C   D
        #
        # A  AA  AB  AC  AD
        # B  BA  BB  BC  BD
        # C  CA  CB  CC  CD
        # D  DA  DB  DC  BB
        # orientation_matrix_defines change of orientation
        #
        fip_axis_0= np.array([[1,1,-1,-1],
                              [1,1,-1,-1],
                              [-1,-1,1,1],
                              [-1,-1,1,1]])
        fip_axis_1= np.array([[1,-1,1,-1],
                              [-1,1,-1,1],
                              [1,-1,1,-1],
                              [-1,1,-1,1]])

        for a1 in range(len(amp_ids)) :
            amp1=amp_ids[a1]
            ii1 = parse_sec_keyword(header['CCDSEC'+amp1])
            a1flux=image[ii1]
            #a1mask=mask[ii1]

            for a2 in range(len(amp_ids)) :
                if a1==a2 :
                    continue
                amp2=amp_ids[a2]
                if cfinder is None : continue
                if not cfinder.haskey("CROSSTALK%s%s"%(amp1,amp2))  : continue
                crosstalk=cfinder.value("CROSSTALK%s%s"%(amp1,amp2))
                if crosstalk==0. : continue
                log.info("Correct for crosstalk=%f from AMP %s into %s"%(crosstalk,amp1,amp2))
                a12flux=crosstalk*a1flux.copy()
                #a12mask=a1mask.copy()
                if fip_axis_0[a1,a2]==-1 :
                    a12flux=a12flux[::-1]
                    #a12mask=a12mask[::-1]
                if fip_axis_1[a1,a2]==-1 :
                    a12flux=a12flux[:,::-1]
                    #a12mask=a12mask[:,::-1]
                ii2 = parse_sec_keyword(header['CCDSEC'+amp2])
                image[ii2] -= a12flux
                # mask[ii2]  |= a12mask (not sure we really need to propagate the mask)

    #- Poisson noise variance (prior to dark subtraction and prior to pixel flat field)
    #- This is biasing, but that's what we have for now
    poisson_var = image.clip(0)

    #- subtract dark after multiplication by gain
    if dark is not False  :
        log.info("subtracting dark")
        image -= trimmed_dark_in_electrons

    #- Correct for dark trails if any
    if not nodarktrail and cfinder is not None :
        for amp in amp_ids :
            if cfinder.haskey("DARKTRAILAMP%s"%amp) :
                amplitude = cfinder.value("DARKTRAILAMP%s"%amp)
                width = cfinder.value("DARKTRAILWIDTH%s"%amp)
                ii    = _parse_sec_keyword(header["CCDSEC"+amp])
                log.info("Removing dark trails for amplifier %s with width=%3.1f and amplitude=%5.4f"%(amp,width,amplitude))
                correct_dark_trail(image,ii,left=((amp=="B")|(amp=="D")),width=width,amplitude=amplitude)

    #- Divide by pixflat image
    pixflat = get_calibration_image(cfinder,"PIXFLAT",pixflat,header)
    if pixflat is not False :
        if pixflat.shape != image.shape:
            raise ValueError('shape mismatch pixflat {} != image {}'.format(pixflat.shape, image.shape))

        almost_zero = 0.001

        if np.all(pixflat > almost_zero ):
            image /= pixflat
            readnoise /= pixflat
            poisson_var /= pixflat**2
        else:
            good = (pixflat > almost_zero )
            image[good] /= pixflat[good]
            readnoise[good] /= pixflat[good]
            poisson_var[good] /= pixflat[good]**2
            mask[~good] |= ccdmask.PIXFLATZERO

        lowpixflat = (0 < pixflat) & (pixflat < 0.1)
        if np.any(lowpixflat):
            mask[lowpixflat] |= ccdmask.PIXFLATLOW




    #- Inverse variance, estimated directly from the data (BEWARE: biased!)
    var = poisson_var + readnoise**2
    ivar = np.zeros(var.shape)
    ivar[var>0] = 1.0 / var[var>0]

    #- Ridiculously high readnoise is bad
    mask[readnoise>100] |= ccdmask.BADREADNOISE

    if bkgsub :
        bkg = _background(image,header)
        image -= bkg

    img = Image(image, ivar=ivar, mask=mask, meta=header, readnoise=readnoise, camera=camera)

    #- update img.mask to mask cosmic rays
    if not nocosmic :
        cosmics.reject_cosmic_rays(img,nsig=cosmics_nsig,cfudge=cosmics_cfudge,c2fudge=cosmics_c2fudge)
        mask = img.mask


    xyset = None

    if model_variance  :

        psf = None
        if psf_filename is None :
            psf_filename = cfinder.findfile("PSF")

        depend.setdep(header, 'CCD_CALIB_PSF', shorten_filename(psf_filename))
        xyset = read_xytraceset(psf_filename)

        fiberflat = None
        with_spectral_smoothing=True
        with_sky_model = True

        if with_sky_model :
            log.debug("Will use a sky model to model the spectra")
            fiberflat_filename = cfinder.findfile("FIBERFLAT")
            depend.setdep(header, 'CCD_CALIB_FIBERFLAT', shorten_filename(fiberflat_filename))
            if fiberflat_filename is not None :
                fiberflat = read_fiberflat(fiberflat_filename)

        log.info("compute an image model after dark correction and pixel flat")
        nsig = 5.
        mimage = compute_image_model(img, xyset, fiberflat=fiberflat,
                                     with_spectral_smoothing=with_spectral_smoothing,
                                     with_sky_model=with_sky_model,
                                     spectral_smoothing_nsig=nsig, psf=psf)

        # here we bring back original image for large outliers
        # this allows to have a correct ivar for cosmic rays and bright sources
        eps = 0.1
        out = (((ivar>0)*(image-mimage)**2/(1./(ivar+(ivar==0))+(0.1*mimage)**2))>nsig**2)
        # out &= (image>mimage) # could request this to be conservative on the variance ... but this could cause other issues
        mimage[out] = image[out]

        log.info("use image model to compute variance")
        if bkgsub :
            mimage += bkg
        if pixflat is not False :
            # undo pixflat
            mimage *= pixflat
        if dark is not False  :
            mimage  += dark
        poisson_var = mimage.clip(0)
        if pixflat is not False :
            if np.all(pixflat > almost_zero ):
                poisson_var /= pixflat**2
            else:
                poisson_var[good] /= pixflat[good]**2
        var = poisson_var + readnoise**2
        ivar[var>0] = 1.0 / var[var>0]

        # regenerate img object
        img = Image(image, ivar=ivar, mask=mask, meta=header, readnoise=readnoise, camera=camera)


    if remove_scattered_light :
        if xyset is None :
            if psf_filename is None :
                psf_filename = cfinder.findfile("PSF")
                depend.setdep(header, 'SCATTERED_LIGHT_PSF', shorten_filename(psf_filename))
            xyset = read_xytraceset(psf_filename)
        img.pix -= model_scattered_light(img,xyset)

    #- Extend header with primary header keywords too
    addkeys(img.meta, primary_header)

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

def recover_2d_dark(hdus,exptime,extname):
    log=get_logger()
    shape=hdus[0].data.shape
    nx=shape[0]
    ny=shape[1]
    profileLeft=hdus[extname].data[0]
    profileRight=hdus[extname].data[1]
    profile_2d_Left=np.transpose(np.tile(profileLeft,(int(ny/2),1)))
    profile_2d_Right=np.transpose(np.tile(profileRight,(int(ny/2),1)))
    profile_2d=np.concatenate((profile_2d_Left,profile_2d_Right),axis=1)
    return profile_2d

def read_dark(filename=None, camera=None, dateobs=None, exptime=None):

    '''
    Return dark current 2D image ( accounting for exposure time)

    Args:

        filename : input filename to read
        camera : e.g. 'b0', 'r1', 'z9'
        dateobs : DATE-OBS string, e.g. '2018-09-23T08:17:03.988'
        exptime : the exposure time of the image in seconds

    Notes:
        must provide filename
    '''
    from astropy.io import fits
    log=get_logger()

    if filename is None:
        #- use camera and dateobs to derive what pixflat file should be used
        raise NotImplementedError

    if exptime is None :
        #- make sure we have the exposure time set to avoid trouble
        raise ValueError("Need exposure time for dark")
    exptime = float(exptime) # will throw exception if cannot cast to float

    hdus = fits.open(filename)
    if len(hdus)==1 :
        log.info("Single dark frame")
        return exptime * hdus[0].data
    else :
        log.info("Exposure time dependent dark")
        exptime_arr=[]
        ext_arr={}
        for hdu in hdus:
            if hdu.header['EXTNAME'] == 'DARK' or hdu.header['EXTNAME'] == 'ZERO':
                pass
            #elif hdu.header['EXTNAME'] == 'ZERO':
            #    ext_arr['0']=hdu.header['EXTNAME']
            #    exptime_arr.append(0)
            else:
                ext_arr[hdu.header['EXTNAME'][1:]]=hdu.header['EXTNAME']
                exptime_arr.append(float(hdu.header['EXTNAME'][1:]))

        exptime_arr = np.array(exptime_arr)
        min_exptime = np.min(exptime_arr)
        max_exptime = np.max(exptime_arr)

        if exptime==0.:
            profile_2d=0.
        elif exptime in exptime_arr:
            profile_2d = recover_2d_dark(hdus,exptime,ext_arr[str(int(exptime))])
        elif exptime < min_exptime :
            log.warning("Use 2D dark profile at min. exptime={}".format(min_exptime))
            profile_2d = recover_2d_dark(hdus,min_exptime,ext_arr[str(int(min_exptime))])
        elif exptime > max_exptime :
            log.warning("Use 2D dark profile at max. exptime={}".format(max_exptime))
            profile_2d = recover_2d_dark(hdus,max_exptime,ext_arr[str(int(max_exptime))])
        else: # Interpolate
            exptime_arr=np.sort(exptime_arr)
            ind=np.where(exptime_arr>exptime)
            ind1=ind[0][0]-1
            ind2=ind[0][0]
            log.info('Interpolate between '+str(exptime_arr[ind1])+' and '+str(exptime_arr[ind2]))
            precision=(exptime-exptime_arr[ind1])/(exptime_arr[ind2]-exptime_arr[ind1])
            # Run interpolation
            image1=recover_2d_dark(hdus,exptime_arr[ind1],ext_arr[str(int(exptime_arr[ind1]))])
            image2=recover_2d_dark(hdus,exptime_arr[ind2],ext_arr[str(int(exptime_arr[ind2]))])
            profile_2d = image1*(1-precision)+image2*precision

        return profile_2d + exptime * hdus['DARK'].data

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
