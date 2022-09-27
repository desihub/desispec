'''
Preprocess raw DESI exposures
'''

import re
import os
import numpy as np
import scipy.interpolate
from pkg_resources import resource_exists, resource_filename
import numba
import time

from scipy import signal
from scipy.ndimage.filters import median_filter
from scipy.signal import fftconvolve

from desispec.image import Image
from desispec import cosmics
from desispec.maskbits import ccdmask
from desiutil.log import get_logger
from desiutil import depend
from desispec.calibfinder import CalibFinder
from desispec.darktrail import correct_dark_trail
from desispec.scatteredlight import model_scattered_light
from desispec.io.xytraceset import read_xytraceset
from desispec.io import read_fiberflat, shorten_filename, findfile
from desispec.io.util import addkeys
from desispec.maskedmedian import masked_median
from desispec.image_model import compute_image_model
from desispec.util import header2night

def get_amp_ids(header):
    '''
    Return list of amp names based upon header keywords
    '''
    amp_ids = []
    for a in ['A', 'B', 'C', 'D', '1', '2', '3', '4'] :
        if 'BIASSEC'+a in header :
            amp_ids.append(a)
    if len(amp_ids)==0 :
        raise KeyError("No keyword BIASSECX with X in A,B,C,D,1,2,3,4 in header")
    return amp_ids

def get_readout_mode(header):
    """
    Derive CCD readout mode from CCD header

    Args:
        header: dict-like FITS header object with BIASSEC keywrds

    Returns "4Amp", "2AmpLeftRight", or "2AmpUpDown"

    "4Amp" means all 4 amps (ABCD) were used for CCD readout;
    "2AmpLeftRight" means 1 left amp (AC) and 1 right amp (BD) were used;
    "2AmpUpDown" means 1 upper amp (CD) and one lower (AB) were used.
    """

    # Amp arrangement:
    #   C D    3 4
    #   A B or 1 2

    # python note: set('ABCD') == set(['A', 'B', 'C', 'D']), not set(['ABCD',])
    ampids = set(get_amp_ids(header))
    if ampids in [set('ABCD'), set('1234')]:
        return "4Amp"
    elif ampids in [set('AB'), set('CD'), set('12'), set('34')]:
        return "2AmpLeftRight"
    elif ampids in [set('AC'), set('BD'), set('13'), set('24')]:
        return "2AmpUpDown"
    else:
        log = get_logger()
        msg = f"Unknown CCD readout mode with amps {ampids}"
        log.error(msg)
        raise ValueError(msg)

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

def compute_overscan_step(overscan_col, median_size=7, edge_margin=50) :
    """
    Compute the overscan step score 'OSTEP' from an array of
    overscan values averaged per CCD row

    Args:
        overscan_col: 1D numpy.array

    Options:
        median_size (int): window size for median pre-filter of overscan_col
        edge_margin (int): ignore this number of rows at the CCD edges

    Returns:
      OSTEP value (float scalar)
    """

    # median filter to futher reduce the noise
    med_overscan_col = median_filter(overscan_col, median_size)

    # use diff. because we want to detect steps, not a continuous variation
    diff_med_overscan_col = np.zeros_like(med_overscan_col)
    diff_med_overscan_col[:-1] = med_overscan_col[1:]-med_overscan_col[:-1]

    # measure the range of variation of overscan while ignoring
    # the edges where we can measure offsets which do not impact the spectroscopy
    diff = diff_med_overscan_col[edge_margin:-edge_margin]
    overscan_step = np.max(diff)-np.min(diff)
    return overscan_step

def _overscan(pix, nsigma=5, niter=3):
    """DEPRECATED: See calc_overscan"""
    log = get_logger()
    log.warning('_overscan is deprecated; please use calc_overscan')
    return calc_overscan(pix, nsigma=nsigma, niter=niter)

def calc_overscan(pix, nsigma=5, niter=3):
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

def subtract_peramp_overscan(image, hdr):
    """Subtract per-amp overscan using BIASSEC* keywords

    Args:
        image: 2D image array, modified in-place
        hdr: FITS header with BIASSEC[ABCD] or BIASSEC[1234] keywords

    Note: currently used in desispec.ccdcalib.compute_bias_file to model
    bias image, but not preproc itself (which subtracts that bias, and
    has more complex support for row-by-row, col-overscan, etc.)
    """
    amp_ids = get_amp_ids(hdr)
    for a,amp in enumerate(amp_ids) :
        ii=parse_sec_keyword(hdr['BIASSEC'+amp])
        s0,s1=ii[0],ii[1]
        for k in ["DATASEC","PRESEC","ORSEC","PRRSEC"] :
            if k+amp in hdr :
                t0,t1=parse_sec_keyword(hdr[k+amp])
                s0 = slice(min(s0.start,t0.start),max(s0.stop,t0.stop))
                s1 = slice(min(s1.start,t1.start),max(s1.stop,t1.stop))
        overscan_image = image[ii].copy()
        overscan,rdnoise = calc_overscan(overscan_image)
        image[s0,s1] -= overscan


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
    bins0=np.linspace(0,image.shape[0],image.shape[0]//patch_width).astype(int)
    bins1=np.linspace(0,image.shape[1],image.shape[1]//patch_width).astype(int)
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

@numba.jit
def numba_mean(image_flux,image_ivar,x,hw=3) :
    """
    Returns mean of pixels vs. row about x+-hw

    Args:
        image_flux: 2D array of CCD image pixels
        image_ivar: 2D array of inverse variance of image_flux
        x: 1D array of x location per row, len(x) = image_flux.shape[0]

    Options:
        hw (int): halfwidth over which to average

    Returns (flux, ivar) 1D arrays with weighted mean and inverse variance of
    pixels[i, int(x-hw):int(x+hw)+1] per row i
    """
    n0=image_flux.shape[0]
    flux=np.zeros(n0)
    ivar=np.zeros(n0)
    for j in range(n0) :
        for i in range(int(x[j]-hw),int(x[j]+hw+1)) :
            flux[j] += image_ivar[j,i]*image_flux[j,i]
            ivar[j] += image_ivar[j,i]
        if ivar[j]>0 :
            flux[j] = flux[j]/ivar[j]
    return flux,ivar


def compute_background_between_fiber_blocks(image,xyset) :
    """
    Computes CCD background between blocks of fibers

    Args:
       image: desispec.image.Image object
       xyset: desispec.xytraceset.XYTraceSet object

    Returns (model, qadict):
       model: np.array of same shape as image.pix
       qadict: dictionary of keywords for QA with min/max per amplifier

    Notes:
        Has hardcoded number of blocks and fibers and typical spacing between
        blocked tuned to DESI spectrographs.
    """

    log = get_logger()
    if 'CAMERA' in image.meta:
        camera = image.meta['CAMERA']
    else:
        camera = 'unknown'

    log.info(f"Camera {camera} estimating CCD background between blocks of fiber traces")

    ivar=image.ivar*(image.mask==0)
    bkg=np.zeros_like(image.pix)

    t0=time.time()

    # first estimate contribution of light from bright fibers

    # inspect only central part of image
    ny=image.pix.shape[0]
    image_yy=np.arange(ny)
    yb=ny//2-300
    ye=ny//2+300

    # we are convolving the image by a lorentzian profile along the cross-dispersion profile
    # as a proxy for the scattered light from bright stars
    # parameters (width of kernel, amplitude, tuned on b7 exposure 117268)
    hw=30
    nw=2*hw+1
    kern=np.zeros((3,nw))
    x=np.arange(-hw,hw+1)
    kern[1]=1/(1+x**2)
    kern *= 0.05/np.sum(kern) # approx normalization
    cimg=fftconvolve(image.pix[yb:ye]*(ivar[yb:ye]>0),kern,mode="same")
    t1=time.time()
    log.info(f"Camera {camera} convolution to estimate contribution of light from bright fibers took {t1-t0:.2f} sec")

    # measure scattered light between blocks
    nblock=21
    scattered_light=np.zeros(nblock)
    for block in range(nblock) :
        # x coordinate of band between fiber blocks
        if block==0 : image_xb =  xyset.x_vs_y(0,image_yy)-7.5
        elif block==20 : image_xb =  xyset.x_vs_y(499,image_yy)+7.5
        else : image_xb = (xyset.x_vs_y(block*25-1,image_yy)+xyset.x_vs_y(block*25,image_yy))/2.
        scattered_light[block] = np.median(numba_mean(cimg,ivar[yb:ye],image_xb[yb:ye]))

    # remove median across interblocks
    scattered_light -= np.median(scattered_light)

    # anything that is higher than 0.5 electron is masked
    # and will be set to the average value of the other blocks (average per amp)
    masked_interblocks=np.where(scattered_light>0.5)[0]

    if masked_interblocks.size>0 :
        log.warning(f"Camera {camera} masking inter blocks {masked_interblocks} because of scattered light from bright fibers")

    qadict = dict()

    for amp in get_amp_ids(image.meta) :
        sec=parse_sec_keyword(image.meta['CCDSEC'+amp])
        log.info(f"Camera {camera} amp {amp} fitting bkg for {sec}")

        # compute value between blocks of fibers
        nblock=21
        xinterblock=[]
        vinterblock=[]
        mwidth=400

        image_yy=np.arange(sec[0].start,sec[0].stop)

        for block in range(nblock) :

            # x coordinate of band between fiber blocks
            if block==0 : image_xb =  xyset.x_vs_y(0,image_yy)-7.5
            elif block==20 : image_xb =  xyset.x_vs_y(499,image_yy)+7.5
            else : image_xb = (xyset.x_vs_y(block*25-1,image_yy)+xyset.x_vs_y(block*25,image_yy))/2.

            # boxcar extraction half width (narrow to avoid pollution by tail of fiber traces for bright stars)
            hw=1
            # use only values in this amplifier (interpolate for others)
            inamp = (image_xb-hw>=sec[1].start)&(image_xb+hw<sec[1].stop)
            if np.all(~inamp) : continue

            if block in masked_interblocks :
                xinterblock.append(image_xb)
                vinterblock.append(np.zeros(image_xb.shape))
                continue

            # extract
            vb,vb_ivar = numba_mean(image.pix[sec[0]],ivar[sec[0]],image_xb)

            # mask out region of brightest sky line in blue camera
            skyline_wave=5578.9
            if block==0 :  skyline_y =  xyset.y_vs_wave(0,skyline_wave)
            elif block==20 : skyline_y =  xyset.y_vs_wave(499,skyline_wave)
            else : skyline_y = (xyset.y_vs_wave(block*25-1,skyline_wave)+xyset.y_vs_wave(block*25,skyline_wave))/2.

            #log.info(f"interblock {block} y({skyline_wave})={skyline_y}")
            inamp &= np.abs(image_yy-skyline_y)>5.

            # keep only in amp values
            if np.any(~inamp) :
                vb = np.interp(image_yy,image_yy[inamp],vb[inamp])

            # median filter
            vb = median_filter(vb,mwidth)

            xinterblock.append(image_xb)
            vinterblock.append(vb)

        xinterblock=np.array(xinterblock)
        vinterblock=np.array(vinterblock)

        #- BBKG = Bundle Background
        vmin, vmax = np.min(vinterblock), np.max(vinterblock)
        qadict['BBKGMIN'+amp] = vmin
        qadict['BBKGMAX'+amp] = vmax

        log.info(f'Camera {camera} amp {amp} interbundle CCD bkg min/max {vmin:.3f} to {vmax:.3f}')

        if np.any(vinterblock!=0) :

            # set average value to masked interblocks
            if np.any(vinterblock==0) :
                vinterblock[vinterblock==0] = np.median(vinterblock[vinterblock>0])

            # interpolate along x
            xx=np.arange(sec[1].start,sec[1].stop)
            for k,y in enumerate(image_yy) :
                bkg[y,sec[1]]=np.interp(xx,xinterblock[:,k],vinterblock[:,k])

    dt = time.time() - t0
    log.info(f"Camera {camera} computing time = {dt:.3f} sec")

    return bkg, qadict

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

    For the case of keyword='BIAS', check for nightly bias before using
    default bias in $DESI_SPECTRO_CALIB
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

        if keyword.upper() == 'BIAS':
            # try biasnight first
            night = header2night(header)
            expid = header['EXPID']
            camera = header['CAMERA'].lower()
            if 'DESI_SPECTRO_REDUX' in os.environ and 'SPECPROD' in os.environ:
                biasnight = findfile('biasnight', night, expid, camera)
                if os.path.exists(biasnight):
                    log.info(f'Using {night} nightly bias for {expid} {camera}')
                    filename = biasnight
                else:
                    log.warning(f'{night} nightly bias not found; using default bias for {expid} {camera}')
            else:
                log.warning(f'SPECPROD not set; using default bias instead of nightly bias for {expid} {camera}')

        if filename is None:
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

def find_overscan_cosmic_trails(rawimage, ov_col, overscan_values, col_width=300,
        threshold=25000., smooth=100):
    """
    Find overscan columns that might be impacted by a trail from bright cosmic

    Args:
        rawimage: numpy 2D array of raw image
        ov_col: tuple(yslice, xslice) from parse_sec_keyword('BIASSECx') defining overscan region

    Options:
        col_width: number of pixels from overscan region to consider
        threshold: ADU threshold for what might cause a problematic trail
        smooth: median filter smoothing scale

    Returns (badrows, active_col_val) where badrows is a boolean array
    of whether each row is bad or not, and active_col_val is an array of
    column-summed and row median-filtered from the active region of the CCD
    next to the overscan region.
    """
    # define a band in the active CCD region next to the overscan
    left_amp = ov_col[1].start < rawimage.shape[1]//2
    if left_amp :
        if ov_col[1].start > rawimage.shape[1]//4 : # overscan is on the right of the active region
            active_col = np.s_[ov_col[0].start:ov_col[0].stop, ov_col[1].start-col_width:ov_col[1].start]
        else : # overscan is on the left of the active region which happens for some 2 amp read mode.
            active_col = np.s_[ov_col[0].start:ov_col[0].stop, ov_col[1].stop:ov_col[1].stop+col_width]
    else :
        active_col = np.s_[ov_col[0].start:ov_col[0].stop, ov_col[1].stop:ov_col[1].stop+col_width]

    # measure sum over columns in band
    active_col_val = np.max(rawimage[active_col].astype(float),axis=1)
    # subtract median filter (to limit effect of neighboring truly bright fiber)
    active_col_val -= median_filter(active_col_val, smooth)
    # flag rows with large signal in active region
    badrows=(active_col_val>threshold)
    med_overscan_col = median_filter(overscan_values, 20)
    badrows &= np.abs(overscan_values-med_overscan_col) > 2.

    # add 2 pixel margins to the list of badrows
    for _ in range(2) :
        badrows[1:] |= badrows[:-1]
        badrows[:-1] |= badrows[1:]

    return badrows, active_col_val

def preproc(rawimage, header, primary_header, bias=True, dark=True, pixflat=True, mask=True,
            bkgsub_dark=False, nocosmic=False, cosmics_nsig=6, cosmics_cfudge=3., cosmics_c2fudge=0.5,
            ccd_calibration_filename=None, nocrosstalk=False, nogain=False,
            overscan_per_row=False, use_overscan_row=False, use_savgol=None,
            nodarktrail=False,remove_scattered_light=False,psf_filename=None,
            bias_img=None,model_variance=False,no_traceshift=False,bkgsub_science=False,
            keep_overscan_cols=False,no_overscan_per_row=False):
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
    Optional background subtraction with median filtering accross the whole CCD if bkgsub_dark=True
    Optional background subtraction with median filtering between groups of fiber traces if bkgsub_science=True

    Optional disabling of cosmic ray rejection if nocosmic=True
    Optional disabling of dark trail correction if nodarktrail=True

    Optional bias image (testing only) may be provided by bias_img=

    Optional tuning of cosmic ray rejection parameters:
        cosmics_nsig: number of sigma above background required
        cosmics_cfudge: number of sigma inconsistent with PSF required
        cosmics_c2fudge:  fudge factor applied to PSF

    Optional fit and subtraction of scattered light

    Optional disabling of overscan subtraction per row if no_overscan_per_row=True

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

    if keep_overscan_cols :
        if dark is not False :
            mess="need dark=False because keep_overscal_col=True, try option --nodark"
            log.error(mess)
            raise RuntimeError(mess)
        if mask is not False :
            mess="need mask=False because keep_overscal_col=True, try option --nomask"
            log.error(mess)
            raise RuntimeError(mess)
        if pixflat is not False :
            mess="need pixflat=False because keep_overscal_col=True, try option --nopixflat"
            log.error(mess)
            raise RuntimeError(mess)

    header = header.copy()
    depend.setdep(header, 'DESI_SPECTRO_CALIB', os.getenv('DESI_SPECTRO_CALIB'))

    for key in ['DESI_SPECTRO_REDUX', 'SPECPROD']:
        if key in os.environ:
            depend.setdep(header, key, os.environ[key])

    cfinder = None
    if ccd_calibration_filename is not False:
        cfinder = CalibFinder([header, primary_header], yaml_file=ccd_calibration_filename)

    #- Check if this file uses amp names 1,2,3,4 (old) or A,B,C,D (new)
    amp_ids = get_amp_ids(header)

    #- if CAMERA is missing, this will raise an exception in a few lines,
    #- but allows CAMERA logging in the meantime if it is present
    try:
        camera = header['CAMERA'].lower()
    except KeyError:
        camera = 'unknown'

    #- Double check that we have the necessary keywords
    missing_keywords = list()
    for key in ['CAMERA', 'EXPID']:
        if key not in header:
            missing_keywords.append(key)

    for prefix in ['CCDSEC', 'BIASSEC']:
        for amp in amp_ids :
            key = prefix+amp
            if not key in header :
                log.error(f'Camera {camera} No {key} keyword in header')
                missing_keywords.append(key)

    if len(missing_keywords) > 0:
        raise KeyError("Camera {} missing keywords {}".format(camera, ' '.join(missing_keywords)))

    #- Subtract bias image

    #- convert rawimage to float64 : this is the output format of read_image
    rawimage = rawimage.astype(np.float64)

    # Savgol
    if cfinder and cfinder.haskey("USE_ORSEC"):
        use_overscan_row = cfinder.value("USE_ORSEC")
    if cfinder and cfinder.haskey("SAVGOL"):
        use_savgol = cfinder.value("SAVGOL")

    # Set bias image, as desired
    if bias_img is None:
        #- will try biasnight first, then default bias
        bias = get_calibration_image(cfinder,"BIAS",bias,header)
    else:
        bias = bias_img

    overscan_col_width = 0

    #- Output arrays
    ny=0
    nx=0
    for amp in amp_ids :
        yy, xx = parse_sec_keyword(header['CCDSEC%s'%amp])
        ny=max(ny,yy.stop)
        nx=max(nx,xx.stop)

    if keep_overscan_cols :
        amp = amp_ids[0]
        tt     = parse_sec_keyword(header['DATASEC'+amp])
        ov_col = parse_sec_keyword(header['BIASSEC%s'%amp])
        overscan_col_width = max((tt[1].start-ov_col[1].start),(ov_col[1].stop-tt[1].stop))
        log.info(f"will keep overscan columns of width = {overscan_col_width} pixels")
        nx += 2*overscan_col_width

    image = np.zeros((ny,nx))

    readnoise = np.zeros_like(image)

    #- Load dark
    if cfinder and cfinder.haskey("DARK") and (dark is not False):

        #- Exposure time
        if cfinder and cfinder.haskey("EXPTIMEKEY") :
            exptime_key=cfinder.value("EXPTIMEKEY")
            log.info(f"Camera {camera} Using exposure time keyword {exptime_key} for dark normalization")
        else :
            exptime_key="EXPTIME"
        exptime =  primary_header[exptime_key]
        log.info(f"Camera {camera} use exptime = {exptime:.1f} sec to compute the dark current")
        
        dark_filename = cfinder.findfile("DARK")
        depend.setdep(header, 'CCD_CALIB_DARK', shorten_filename(dark_filename))
        log.info(f'Camera {camera} using DARK model from {dark_filename}')
        # dark is multipled by exptime, or we use the non-linear dark model in the routine
        dark = read_dark(filename=dark_filename,exptime=exptime)

        if dark.shape == image.shape :
            log.info(f"Camera {camera} dark is trimmed")
            trimmed_dark_in_electrons = dark
            dark_is_trimmed   = True
        elif dark.shape == rawimage.shape :
            log.info(f"Camera {camera} dark is not trimmed")
            trimmed_dark_in_electrons = np.zeros_like(image)
            dark_is_trimmed = False
        else :
            message="Camera {} incompatible dark shape={} when raw shape={} and preproc shape={}".format(
                    camera, dark.shape, rawimage.shape, image.shape)
            log.error(message)
            raise ValueError(message)

        if np.all(dark==0.0):
            if exptime == 0.0:
                log.info(f'Camera {camera} dark model for exptime=0 is all zeros; not applying')
            else:
                log.error(f'Camera {camera} dark model for exptime={exptime} unexpectedly all zeros; not applying')
            dark = False

    else:
        dark = False

    if bias is not False : #- it's an array
        if bias.shape == rawimage.shape  :
            log.info(f"Camera {camera} subtracting bias")
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


    if no_overscan_per_row :
        log.debug("Option no_overscan_per_row is set")

    for amp in amp_ids:
        # Grab the sections
        ov_col = parse_sec_keyword(header['BIASSEC'+amp])
        if 'ORSEC'+amp in header.keys():
            ov_row = parse_sec_keyword(header['ORSEC'+amp])
        elif use_overscan_row:
            log.error(f'Camera {camera} no ORSEC{amp} keyword; not using overscan_row')
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
                    log.info(f'Camera {camera} using GAIN{amp}={gain} from calibration data')
                else :
                    gain = 1.0
                    log.error(f'Camera {camera} missing keyword GAIN{amp} in header and nothing in calib data; using {gain}')

        #- Record what gain value was actually used
        header['GAIN'+amp] = gain

        #- Add saturation level
        if 'SATURLEV'+amp in header:
            saturlev_adu = header['SATURLEV'+amp]          # in ADU
        else:
            if cfinder and cfinder.haskey('SATURLEV'+amp) :
                saturlev_adu = float(cfinder.value('SATURLEV'+amp))
                log.info(f'Camera {camera} using SATURLEV{amp}={saturlev_adu} from calibration data')
            else :
                saturlev_adu = 2**16-1 # 65535 is the max value in the images
                log.warning(f'Camera {camera} Missing keyword SATURLEV{amp} in header and nothing in calib data; using {saturlev_adu} ADU')
        header['SATULEV'+amp] = (saturlev_adu,"saturation or non lin. level, in ADU, inc. bias")

        # Generate the overscan images
        raw_overscan_col = rawimage[ov_col].copy()

        kk = parse_sec_keyword(header['CCDSEC'+amp])

        if keep_overscan_cols :
            if kk[1].stop>image.shape[1]//2 :
                start = kk[1].start + overscan_col_width
                stop  = kk[1].stop + 2*overscan_col_width
            else :
                start = kk[1].start
                stop  = kk[1].stop + overscan_col_width
            kk = np.s_[kk[0].start:kk[0].stop, start:stop]

        # Now remove the overscan_col
        nrows=raw_overscan_col.shape[0]
        log.info(f"Camera {camera} {nrows} rows in overscan")

        if not nodarktrail and cfinder is not None :
            if cfinder.haskey("DARKTRAILAMP%s"%amp) :
                log.info("Perform a dark trail correction before fitting the overscan region")
                amplitude = cfinder.value("DARKTRAILAMP%s"%amp)
                width = cfinder.value("DARKTRAILWIDTH%s"%amp)
                # region is BIASSEC+DATASEC
                ii    = parse_sec_keyword(header["BIASSEC"+amp])
                jj    = parse_sec_keyword(header["DATASEC"+amp])
                start = min(ii[1].start,jj[1].start)
                stop  = max(ii[1].stop,jj[1].stop)
                jj=np.s_[jj[0].start:jj[0].stop, start:stop]
                o,r = calc_overscan(rawimage[jj])
                # tmp copy or rawimage
                tmp=rawimage[jj].copy()-o
                ll=np.s_[0:tmp.shape[0],0:tmp.shape[1]]
                correct_dark_trail(tmp,ll,left=((amp=="B")|(amp=="D")),width=width,amplitude=amplitude)
                tmp -= (rawimage[jj].copy()-o) # subtract input to keep only the correction
                start = ii[1].start-jj[1].start
                stop  = ii[1].stop-jj[1].start
                raw_overscan_col += tmp[:,start:stop] # apply the correction only to the overscan cols

        overscan_col = np.zeros(nrows)
        rdnoise  = np.zeros(nrows)
        for j in range(nrows) :
            if np.isnan(np.sum(overscan_col[j])) :
                log.warning(f"Camera {camera} amp {amp} NaN values in row {j} of overscan")
                continue
            o,r =  calc_overscan(raw_overscan_col[j])
            overscan_col[j]=o
            rdnoise[j]=r

        # find rows impacted by a large cosmic charge deposit
        badrows, active_col_val = find_overscan_cosmic_trails(rawimage, ov_col, overscan_values = overscan_col)
        if np.any(badrows) :
            log.warning("Camera {} amp {}, ignore overscan rows = {} because of large charge deposit = {} ADUs".format(
                camera,amp,np.where(badrows)[0],active_col_val[badrows]))
            # do not use overscan value for those, use interpolation
            goodrows = ~badrows
            rr=np.arange(nrows)
            try:
                overscan_col[badrows] = np.interp(rr[badrows],rr[goodrows],overscan_col[goodrows])
            except ValueError:
                # If can't interpolate, log error but don't crash and let ostep do the flagging
                ngood = np.sum(goodrows)
                nbad = np.sum(badrows)
                log.error(f'Camera {camera} amp {amp} unable to interpolate overscan_col over {nbad} bad rows using {ngood} good rows')

        overscan_step = compute_overscan_step(overscan_col)
        header['OSTEP'+amp] = (overscan_step,'ADUs (max-min of median overscan per row)')
        log.info(f"Camera {camera} amp {amp} overscan max-min per row (OSTEP) = {overscan_step:2f} ADU")
        if overscan_step <  2 or no_overscan_per_row : # tuned to trig on the worst few
            log.info(f"Camera {camera} amp {amp} subtracting average overscan")
            o,r =  calc_overscan(raw_overscan_col)
            # replace by single value
            overscan_col = np.repeat(o,nrows)
            rdnoise  = np.repeat(r,nrows)
            header['OMETH'+amp]=("AVERAGE","use average overscan")
        else :
            header['OMETH'+amp]=("PER_ROW","use average overscan per row")
            log.info(f"Camera {camera} amp {amp} subtracting overscan per row")

            # The threshold of 5 ADUs discards 20% of the r8-A data
            # from Oct 2021. But it is necessary to discard some
            # exposures with OSTEPA>=8 ADU where bias variation
            # residuals were the cause of some bad redshifts.
            if overscan_step > 5. :
                mask[kk] |= ccdmask.BADREADNOISE
                log.warning(f"Camera {camera} amp {amp} OSTEP={overscan_step:.2f} is too large, set ccdmask.BADREADNOISE bit mask")

            # We use the overscan per row but we still compute a single readnoise value for the whole amplifier
            row_subtracted_overscan_col = raw_overscan_col - overscan_col[:,None]
            o,r = calc_overscan(row_subtracted_overscan_col)
            rdnoise  = np.repeat(r,nrows)
        if bias is not False :
            # the master bias noise is already in the raw data
            # (because we already subtracted the bias)
            # so we only need to add the quadratic difference of the master bias read noise
            # between the active region and the overscan columns
            jj = parse_sec_keyword(header['DATASEC'+amp])

            o,biasnoise_datasec = calc_overscan(bias[jj])
            o,biasnoise_ovcol   = calc_overscan(bias[ov_col])
            new_rdnoise         = np.sqrt(rdnoise**2+biasnoise_datasec**2-biasnoise_ovcol**2)
            log.info("Camera {} amp {} master bias noise {:4.3f} ADU, rdnoise {:4.3f} -> {:4.3f} ADU".format(
                camera, amp, biasnoise_datasec, np.mean(rdnoise), np.mean(new_rdnoise)))
            rdnoise = new_rdnoise

        if use_overscan_row:
            raw_overscan_row = rawimage[ov_row].copy()
            # Remove overscan_col from overscan_row
            o,r = calc_overscan(raw_overscan_col)
            overscan_row = raw_overscan_row - o

        rdnoise *= gain
        median_rdnoise  = np.median(rdnoise)
        median_overscan = np.median(overscan_col)
        log.info(f"Camera {camera} amp {amp} Median rdnoise and overscan= {median_rdnoise:.3f} {median_overscan:.3f}")

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
                log.error('Camera {} amp {} measured readnoise {:.2f} < 0.5 * expected readnoise {:.2f}'.format(
                    camera, amp, median_rdnoise, expected_readnoise))
            elif median_rdnoise < 0.9*expected_readnoise:
                log.warning('Camera {} amp {} measured readnoise {:.2f} < 0.9 * expected readnoise {:.2f}'.format(
                    camera, amp, median_rdnoise, expected_readnoise))
            elif median_rdnoise > 2.0*expected_readnoise:
                log.error('Camera {} amp {} measured readnoise {:.2f} > 2 * expected readnoise {:.2f}'.format(
                    camera, amp, median_rdnoise, expected_readnoise))
            elif median_rdnoise > 1.2*expected_readnoise:
                log.warning('Camera {} amp {} measured readnoise {:.2f} > 1.2 * expected readnoise {:.2f}'.format(
                    camera, amp, median_rdnoise, expected_readnoise))

        log.info("Camera {} amp {} measured readnoise = {:.3f}".format(
            camera, amp, median_rdnoise))

        #- subtract overscan from data region and apply gain
        jj = parse_sec_keyword(header['DATASEC'+amp])
        if keep_overscan_cols :
            start=min(jj[1].start,ov_col[1].start)
            stop=max(jj[1].stop,ov_col[1].stop)
            jj = np.s_[jj[0].start:jj[0].stop, start:stop]

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
                log.info(f"Camera {camera} amp {amp} Using savgol")
                collapse_oscan_row = np.zeros(overscan_row.shape[1])
                for col in range(overscan_row.shape[1]):
                    o, _ = calc_overscan(overscan_row[:,col])
                    collapse_oscan_row[col] = o
                oscan_row = _savgol_clipped(collapse_oscan_row, niter=0)
                oimg_row = np.outer(np.ones(data.shape[0]), oscan_row)
                data -= oimg_row
            else:
                o,r = calc_overscan(overscan_row)
                log.info("Camera {} amp {} removing overscan rows value = {:.2f}".format(camera,amp,o))
                data -= o

        #- apply saturlev (defined in ADU), prior to multiplication by gain
        saturated = (rawimage[jj]>=saturlev_adu)
        mask[kk][saturated] |= ccdmask.SATURATED

        #- ADC to electrons
        image[kk] = data*gain

        #- add Poisson noise of bias
        if ( bias is not False ) and ( bias is not None ) :
            trimmed_bias_in_electrons = bias[jj]*gain
            readnoise[kk] = np.sqrt(readnoise[kk]**2+trimmed_bias_in_electrons*(trimmed_bias_in_electrons>0))

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
                log.info("Camera {} correct for crosstalk={} from AMP {} into {}".format(
                    camera, crosstalk, amp1, amp2))
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
        log.info(f"Camera {camera} subtracting dark")
        image -= trimmed_dark_in_electrons
        # measure its noise
        new_readnoise = np.zeros(readnoise.shape)
        for amp in amp_ids:
            kk = parse_sec_keyword(header['CCDSEC'+amp])
            o,darknoise = calc_overscan(trimmed_dark_in_electrons[kk])
            new_readnoise[kk] = np.sqrt(readnoise[kk]**2+darknoise**2)
            log.info("Camera {} amp {} master dark noise = {:4.3f} elec, rdnoise {:4.3f} -> {:4.3f} elec".format(
                camera, amp, darknoise, np.mean(readnoise[kk]), np.mean(new_readnoise[kk])))
        readnoise = new_readnoise

    #- Correct for dark trails if any
    if not nodarktrail and cfinder is not None :
        for amp in amp_ids :
            if cfinder.haskey("DARKTRAILAMP%s"%amp) :
                amplitude = cfinder.value("DARKTRAILAMP%s"%amp)
                width = cfinder.value("DARKTRAILWIDTH%s"%amp)
                ii    = parse_sec_keyword(header["CCDSEC"+amp])
                if keep_overscan_cols and ii[1].stop>image.shape[1]//2 :
                    start = ii[1].start + overscan_col_width
                    stop  = ii[1].stop  + 2*overscan_col_width
                    ii = np.s_[ii[0].start:ii[0].stop, start:stop]
                else :
                    start = ii[1].start
                    stop  = ii[1].stop  + overscan_col_width
                    ii = np.s_[ii[0].start:ii[0].stop, start:stop]
                log.info("Camera {} amp {} removing dark trails with width={:3.1f} and amplitude={:5.4f}".format(
                    camera, amp, width, amplitude))
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

    #- High readnoise is bad
    mask[readnoise>15] |= ccdmask.BADREADNOISE

    if bkgsub_dark :
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
            log.debug(f"Camera {camera} will use a sky model to model the spectra")
            fiberflat_filename = cfinder.findfile("FIBERFLAT")
            depend.setdep(header, 'CCD_CALIB_FIBERFLAT', shorten_filename(fiberflat_filename))
            if fiberflat_filename is not None :
                fiberflat = read_fiberflat(fiberflat_filename)

        log.info(f"Camera {camera} compute an image model after dark correction and pixel flat")
        nsig = 5.
        mimage = compute_image_model(img, xyset, fiberflat=fiberflat,
                                     with_spectral_smoothing=with_spectral_smoothing,
                                     with_sky_model=with_sky_model,
                                     spectral_smoothing_nsig=nsig, psf=psf, fit_x_shift=(not no_traceshift))

        # here we bring back original image for large outliers
        # this allows to have a correct ivar for cosmic rays and bright sources
        eps = 0.1
        out = (((ivar>0)*(image-mimage)**2/(1./(ivar+(ivar==0))+(0.1*mimage)**2))>nsig**2)
        # out &= (image>mimage) # could request this to be conservative on the variance ... but this could cause other issues
        mimage[out] = image[out]

        log.info(f"Camera {camera} use image model to compute variance")
        if bkgsub_dark :

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

    if np.all(img.mask>0):
        log.error(f'Camera {camera} is entirely masked (i.e. unusable)')

    if remove_scattered_light :
        if xyset is None :
            if psf_filename is None :
                psf_filename = cfinder.findfile("PSF")
                depend.setdep(header, 'SCATTERED_LIGHT_PSF', shorten_filename(psf_filename))
            xyset = read_xytraceset(psf_filename)
        img.pix -= model_scattered_light(img,xyset)

    if bkgsub_science :
        if xyset is None :
            if psf_filename is None :
                psf_filename = cfinder.findfile("PSF")
                depend.setdep(header, 'SCATTERED_LIGHT_PSF', shorten_filename(psf_filename))
            xyset = read_xytraceset(psf_filename)
        ccdbkg, bkgqa = compute_background_between_fiber_blocks(img,xyset)
        img.pix -= ccdbkg

        #- Adds BBKG (Bundle Background) MIN/MAX per amp for QA/debugging
        addkeys(img.meta, bkgqa)

    #- Extend header with primary header keywords too
    try:
        img.meta.extend(primary_header, strip=True, unique=True)
    except AttributeError:
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
