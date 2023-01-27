'''
desispec.image_model
====================

Function to provide rapidly a model of a preprocessed image
in order to have a pixel variance estimation that is
almost not correlated with pixel value.
'''


import os
import numpy as np
import copy
import time

import numba
import scipy.interpolate

from desispec.image import Image
from desiutil.log import get_logger
from desispec.io.xytraceset import read_xytraceset
from desispec.qproc.qextract import qproc_boxcar_extraction
from desispec.qproc.qsky import qproc_sky_subtraction
from desispec.qproc.qfiberflat import qproc_apply_fiberflat
from desispec.trace_shifts import compute_dx_from_cross_dispersion_profiles

@numba.jit
def numba_proj(image,x,sigma,flux) :
    '''
    Add a spectrum to a model of the pixel values in a CCD image assuming a Gaussian cross-dispersion profile.

    Args:
       image: 2D numpy array , this array will be modified
       x: 1D numpy array of size image.shape[0], coordinate of center of cross-dispersion profile of spectral trace, in pixel units, for each CCD row
       sigma : 1D numpy array of size image.shape[0], sigma of cross-dispersion profile of spectral trace, in pixel units, for each CCD row
       flux : 1D numpy array of size image.shape[0], quantity to project (for design usage: electrons per pixel) for each CCD row
    '''
    n0=image.shape[0]
    n1=image.shape[1]
    hw=3
    prof=np.zeros(2*hw+1)
    for j in range(n0) :
        sumprof=0.
        b=max(0,int(x[j]-hw))
        e=min(n1,int(x[j]+hw+1))
        for i in range(b,e) :
            prof[i-b]=np.exp(-(i-x[j])**2/2./sigma[j]**2)
            sumprof += prof[i-b]
        for i in range(b,e) :
            image[j,i] += flux[j]*prof[i-b]/sumprof



def compute_image_model(image,xytraceset,fiberflat=None,fibermap=None,with_spectral_smoothing=True,with_sky_model=True,\
                        spectral_smoothing_sigma_length=10.,spectral_smoothing_nsig=4.,psf=None,fit_x_shift=True):
    '''
    Returns a model of the input image, using a fast extraction, a processing of
    spectra with a common sky model and a smoothing, followed by a reprojection
    on the CCD image.

    Args:
        image: a preprocessed image in the form of a desispec.image.Image object
        xytraceset: a desispec.xytraceset.XYTraceSet object with trace coordinates
        fiberflat, optional: a desispec.fiberflat.FiberFlat object
        with_spectral_smoothing, optional: try and smooth the spectra to reduce noise (and eventualy reduce variance correlation)
        with_sky_model, optional: use a sky model as part of the spectral modeling to reduce the noise (requires a fiberflat)
        spectral_smoothing_sigma_length, optional: sigma of Gaussian smoothing along wavelength in A
        spectral_smoothing_nsig, optional: number of sigma rejection threshold to fall back to the original extracted spectrum instead of the smooth one
        psf, optional specter.psf.GaussHermitePSF object to be used for the 1D projection (slow, by default=None, in which case a Gaussian profile is used)
        fit_x_shift, optional: fit for an offset of the spectral traces.

    Returns:
        a 2D np.array of same shape as image.pix
    '''

    log=get_logger()

    if fit_x_shift :
        t0=time.time()
        log.info("fitting dx ...")
        x,y,dx,ex,fiber,wave = compute_dx_from_cross_dispersion_profiles(xcoef=xytraceset.x_vs_wave_traceset._coeff,
                                                                         ycoef=xytraceset.y_vs_wave_traceset._coeff,
                                                                         wavemin=xytraceset.wavemin,
                                                                         wavemax=xytraceset.wavemax,
                                                                         image=image,
                                                                         fibers=np.arange(xytraceset.nspec,dtype=int))
        dx = np.median(dx)
        log.info("measured trace shift dx = {:.3f} pixel".format(dx))
        log.info("dx fit took {:.2f} sec".format(time.time()-t0))
        xytraceset.x_vs_wave_traceset._coeff[:,0] += dx

    # first perform a fast boxcar extraction
    log.info("extract spectra")
    image.mask = None
    qframe = qproc_boxcar_extraction(xytraceset,image)
    fqframe = None
    sqframe = None
    ratio = None
    if fiberflat is not None :
        log.info("fiberflat")
        fqframe = copy.deepcopy(qframe)
        flat=qproc_apply_fiberflat(fqframe,fiberflat=fiberflat,return_flat=True)
    if with_sky_model :
        if fiberflat is None :
            log.warning("cannot compute and use a sky model without a fiberflat")
        else :
            # crude model of the sky, accounting for fiber throughput variation
            log.info("sky")
            sqframe = copy.deepcopy(fqframe)
            sky = qproc_sky_subtraction(sqframe,return_skymodel=True)

    if with_spectral_smoothing :

        log.info("spectral smoothing")
        sigma = spectral_smoothing_sigma_length/np.mean(np.gradient(qframe.wave[qframe.nspec//2]))
        log.debug("smoothing sigma in flux bin units = {}".format(sigma))
        hw=int(3*sigma)
        u=(np.arange(2*hw+1)-hw)
        kernel=np.exp(-u**2/sigma**2/2.)
        kernel/=np.sum(kernel)
        nsig=5

        y=np.arange(qframe.flux.shape[1])
        for s in range(qframe.nspec) :
            if sqframe is not None :
                fflux=sqframe.flux[s]
                fivar=sqframe.ivar[s]
            elif fqframe is not None :
                fflux=fqframe.flux[s]
                fivar=fqframe.ivar[s]
            else :
                fflux=qframe.flux[s]
                fivar=qframe.ivar[s]
            sfflux=scipy.signal.fftconvolve(fflux,kernel,"same")
            good=((fivar*(fflux-sfflux)**2)<(spectral_smoothing_nsig**2))
            out=~good
            nout=np.sum(out)
            if nout>0 :
                # recompute sflux while masking outliers
                tflux = fflux.copy()
                tflux[out] = np.interp(y[out],y[good],fflux[good])
                sfflux=scipy.signal.fftconvolve(tflux,kernel,"same")
                # and replace the 'out' region by the original data
                # because we want to keep it to have a fair variance
                sfflux[out] = fflux[out]

            # replace by smooth version (+ possibly average sky)
            if sqframe is not None :
                qframe.flux[s]=(sky[s]+sfflux)*flat[s]
            elif fqframe is not None :
                qframe.flux[s]=sfflux*flat[s]
            else :
                qframe.flux[s]=sfflux

    log.info("project back spectra on image")
    y=np.arange(image.pix.shape[0])

    # cross dispersion profile
    xsig=1.*np.ones((qframe.nspec,image.pix.shape[0]))
    if xytraceset.xsig_vs_wave_traceset :
        log.info("use traceset in PSF for xsig")
        for s in range(xytraceset.nspec) :
            wave = xytraceset.wave_vs_y(s,y)
            xsig[s] = xytraceset.xsig_vs_wave(s,wave)
    else :
        log.info("use default xsig={}".format(np.mean(xsig)))
    # keep only positive flux
    qframe.flux *= (qframe.flux>0.)

    # convert counts per A to counts per pixel
    dwave = np.gradient(qframe.wave,axis=1)
    qframe.flux *= dwave

    # because true profile is not Gaussian and we do not integrate the
    # profile in pixels, we have to apply an adjustment
    empirical_scale = 1.1
    log.info("empirical adjustment of xsig by {:4.2f}".format(empirical_scale))
    xsig *= empirical_scale

    model=np.zeros(image.pix.shape)

    if psf is None : # use simple Gaussian
        log.info("use Gaussian sigma of average = {:4.2f}".format(np.mean(xsig)))
        for s in range(qframe.nspec) :
            x=xytraceset.x_vs_y(s,y)
            numba_proj(model,x,xsig[s],qframe.flux[s])
    else : # this takes a lot of time
        log.debug("Use PSF for projection, but in 1D")
        psf._polyparams['HSIZEY']=0
        psf._polyparams['GHDEGY']=0
        model = psf.project(qframe.wave, qframe.flux, xyrange=(0,image.pix.shape[1],0,image.pix.shape[0]))
        log.debug("done projecting")

    log.info("done")
    return model
