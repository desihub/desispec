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
from desispec.calibfinder import CalibFinder
from desispec.darktrail import correct_dark_trail

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
        overscan (float):
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


def _savgol_clipped(data, window=165, polyorder=5, niter=3, threshold=3.):
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

def masked_median(images,masks=None) :
    '''
    Perfomes a median of an list of input images. If a list of mask is provided,
    the median is performed only on unmasked pixels.

    Args:
       images : 3D numpy array : list of images of same shape
    Options:
       masks : list of mask images of same shape as the images. Only pixels with mask==0 are considered in the median.

    Returns : median image
    '''
    log = get_logger()

    if masks is None :
        log.info("simple median of %d images"%len(images))
        return np.median(images,axis=0)
    else :
        log.info("masked array median of %d images"%len(images))
        return np.ma.median(np.ma.masked_array(data=images,mask=(masks!=0)),axis=0).data

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

def get_calibration_image(cfinder,keyword,entry) :
    """Please provide documentation for this function!
    """
    log=get_logger()

    if entry is False : return False # we don't want do anything


    filename = None
    if entry is True :
        # we have to find the filename
        if cfinder is None :
            log.error("no calibration data was found")
            raise ValueError("no calibration data was found")
        if cfinder.haskey(keyword) :
            filename = cfinder.findfile(keyword)
        else :
            return False # we say in the calibration data we don't need this
    elif isinstance(entry,str) :
        filename = entry
    else :
        return entry # it's expected to be an image array


    log.info("Using %s %s"%(keyword,filename))
    if keyword == "BIAS" :
        return read_bias(filename=filename)
    elif keyword == "MASK" :
        return read_mask(filename=filename)
    elif keyword == "DARK" :
        return read_dark(filename=filename)
    elif keyword == "PIXFLAT" :
        return read_pixflat(filename=filename)
    else :
        log.error("Don't known how to read %s in %s"%(keyword,path))
        raise ValueError("Don't known how to read %s in %s"%(keyword,path))
    return False

def preproc(rawimage, header, primary_header, bias=True, dark=True, pixflat=True, mask=True,
            bkgsub=False, nocosmic=False, cosmics_nsig=6, cosmics_cfudge=3., cosmics_c2fudge=0.5,
            ccd_calibration_filename=None, nocrosstalk=False, nogain=False,
            overscan_per_row=False, use_overscan_row=True, flag_savgol=None,
            nodarktrail=False):

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
            from the data (default: True).  Requires ORSEC in
            the Header
        flag_savgol : bool,  Specify whether to use Savitsky-Golay filter for
            the overscan.  If not set and cfinder is not initialized, the
            default is True.

    Optional background subtraction with median filtering if bkgsub=True

    Optional disabling of cosmic ray rejection if nocosmic=True
    Optional disabling of dark trail correction if nodarktrail=True

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
    log=get_logger()

    header = header.copy()
    
    cfinder = None
    
    if ccd_calibration_filename is not False:
        cfinder = CalibFinder([header, primary_header], yaml_file=ccd_calibration_filename)

    #- TODO: Check for required keywords first

    #- Subtract bias image
    camera = header['CAMERA'].lower()

    #- convert rawimage to float64 : this is the output format of read_image
    rawimage = rawimage.astype(np.float64)

    # Savgol
    use_savgol = True
    if cfinder and cfinder.haskey("SAVGOL"):
        use_savgol = cfinder.value("SAVGOL")
    # Over-ride savgol?
    if flag_savgol is not None:
        use_savgol = flag_savgol

    import pdb; pdb.set_trace()

    bias = get_calibration_image(cfinder,"BIAS",bias)

    if bias is not False : #- it's an array
        if bias.shape == rawimage.shape  :
            log.info("subtracting bias")
            rawimage = rawimage - bias
        else:
            raise ValueError('shape mismatch bias {} != rawimage {}'.format(bias.shape, rawimage.shape))

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

    #- Load mask
    mask = get_calibration_image(cfinder,"MASK",mask)

    if mask is False :
        mask = np.zeros(image.shape, dtype=np.int32)
    else :
        if mask.shape != image.shape :
            raise ValueError('shape mismatch mask {} != image {}'.format(mask.shape, image.shape))

    #- Load dark
    dark = get_calibration_image(cfinder,"DARK",dark)

    if dark is not False :
        if dark.shape != image.shape :
            log.error('shape mismatch dark {} != image {}'.format(dark.shape, image.shape))
            raise ValueError('shape mismatch dark {} != image {}'.format(dark.shape, image.shape))


        if cfinder and cfinder.haskey("EXPTIMEKEY") :
            exptime_key=cfinder.value("EXPTIMEKEY")
            log.info("Using exposure time keyword %s for dark normalization"%exptime_key)
        else :
            exptime_key="EXPTIME"
        exptime =  primary_header[exptime_key]

        log.info("Multiplying dark by exptime %f"%(exptime))
        dark *= exptime

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


        #- Add saturation level
        if 'SATURLEV'+amp in header:
            saturlev = header['SATURLEV'+amp]          # in electrons
        else:
            if cfinder and cfinder.haskey('SATURLEV'+amp) :
                saturlev = float(cfinder.value('SATURLEV'+amp))
                log.info('Using SATURLEV{}={} from calibration data'.format(amp,saturlev))
            else :
                saturlev = 200000
                log.warning('Missing keyword SATURLEV{} in header and nothing in calib data; using 200000'.format(amp,saturlev))

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
        # And now the rows
        if use_overscan_row:
            # Savgol?
            if use_savgol:
                collapse_oscan_row = np.zeros(overscan_row.shape[1])
                for col in range(overscan_row.shape[1]):
                    o, _ = _overscan(overscan_row[:,col])
                    collapse_oscan_row[col] = o
                oscan_row = _savgol_clipped(collapse_oscan_row)
                oimg_row = np.outer(np.ones(data.shape[0]), oscan_row)
                data -= oimg_row
            else:
                o,r = _overscan(overscan_row)
                data -= o

        #- apply saturlev (defined in ADU), prior to multiplication by gain
        saturated = (rawimage[jj]>=saturlev)
        mask[kk][saturated] |= ccdmask.SATURATED

        #- subtract dark prior to multiplication by gain
        if dark is not False  :
            log.info("subtracting dark for amp %s"%amp)
            data -= dark[kk]

        image[kk] = data*gain

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
    pixflat = get_calibration_image(cfinder,"PIXFLAT",pixflat)
    if pixflat is not False :
        if pixflat.shape != image.shape:
            raise ValueError('shape mismatch pixflat {} != image {}'.format(pixflat.shape, image.shape))

        almost_zero = 0.001

        if np.all(pixflat > almost_zero ):
            image /= pixflat
            readnoise /= pixflat
        else:
            good = (pixflat > almost_zero )
            image[good] /= pixflat[good]
            readnoise[good] /= pixflat[good]
            mask[~good] |= ccdmask.PIXFLATZERO

        lowpixflat = (0 < pixflat) & (pixflat < 0.1)
        if np.any(lowpixflat):
            mask[lowpixflat] |= ccdmask.PIXFLATLOW

    #- Inverse variance, estimated directly from the data (BEWARE: biased!)
    var = image.clip(0) + readnoise**2
    ivar = np.zeros(var.shape)
    ivar[var>0] = 1.0 / var[var>0]

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
