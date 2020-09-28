'''
Function to provide rapidly a model of a preprocessed image
in order to have a pixel variance estimation that is
almost not correlated with pixel value.
'''


import os
import numpy as np
import copy

import numba
import scipy.interpolate

from desispec.image import Image
from desiutil.log import get_logger
from desispec.io.xytraceset import read_xytraceset
from desispec.qproc.qextract import qproc_boxcar_extraction
from desispec.qproc.qsky import qproc_sky_subtraction
from desispec.qproc.qfiberflat import qproc_apply_fiberflat

@numba.jit
def numba_proj(image,x,sigma,flux) :
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



def compute_image_model(image,xytraceset,fiberflat=None,fibermap=None,with_spectral_smoothing=True,with_sky_model=True):
    '''
    Returns a model of the input image, using a fast extraction, a processing of
    spectra with a common sky model and a smoothing, followed by a reprojection
    on the CCD image.

    Inputs:
       image: a preprocessed image in the form of a desispec.image.Image object
       xytraceset: a desispec.xytraceset.XYTraceSet object with trace coordinates

    Optional:
       fiberflat: a desispec.fiberflat.FiberFlat object
       with_spectral_smoothing: try and smooth the spectra to reduce noise (and eventualy reduce variance correlation)
       with_sky_model: use a sky model as part of the spectral modeling to reduce the noise (requires a fiberflat)

    returns:
       a 2D np.array of same shape as image.pix
    '''

    log=get_logger()

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
        sigma = 20.
        hw=int(3*sigma)
        u=(np.arange(2*hw+1)-hw)
        kernel=np.exp(-u**2/sigma**2/2.)
        kernel/=np.sum(kernel)
        nsig=3.

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
            good=((fivar*(fflux-sfflux)**2)<(nsig**2))
            out=~good
            nout=np.sum(out)
            if nout>0 :
                #if nout>50: log.warning("for spectrum={} number of pixels not modeled for variance={}".format(s,np.sum(out)))
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
        for s in range(xytraceset.nspec) :
            wave = xytraceset.wave_vs_y(s,y)
            xsig[s] = xytraceset.xsig_vs_wave(s,wave)

    # keep only positive flux
    qframe.flux *= (qframe.flux>0.)

    model=np.zeros(image.pix.shape)
    for s in range(qframe.nspec) :
        x=xytraceset.x_vs_y(s,y)
        numba_proj(model,x,xsig[s],qframe.flux[s])

    #import astropy.io.fits as pyfits
    #pyfits.writeto("model.fits",model,overwrite=True)
    #log.warning("WRITING A DEBUG FILE model.fits")

    log.info("done")
    return model
