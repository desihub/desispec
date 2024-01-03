"""
desispec.qproc.qextract
=======================

"""
import time
import numpy as np
from numpy.polynomial.legendre import legval
import numba

from desiutil.log import get_logger
from desispec.xytraceset import XYTraceSet
from desispec.image import Image
from desispec.io.fibermap import empty_fibermap
from desispec.qproc.qframe import QFrame


@numba.jit(nopython=True)
def numba_extract(image_flux,image_var,x,hw=3) :
    n0=image_flux.shape[0]
    flux=np.zeros(n0)
    ivar=np.zeros(n0)
    for j in range(n0) :
        var=0
        for i in range(int(x[j]-hw),int(x[j]+hw+1)) :
            flux[j] += image_flux[j,i]
            if image_var[j,i]>0 :
                var += image_var[j,i]
            else :
                flux[j]=0
                ivar[j]=0
                var=0
                break
        if var>0 :
            ivar[j] = 1./var
    return flux,ivar



def qproc_boxcar_extraction(xytraceset, image, fibers=None, width=7, fibermap=None, save_sigma=True) :
    """
    Fast boxcar extraction of spectra from a preprocessed image and a trace set

    Args:
        xytraceset : DESI XYTraceSet object
        image : DESI preprocessed Image object

    Optional:
        fibers : 1D np.array of int (default is all fibers, the first fiber is always = 0)
        width  : extraction boxcar width, default is 7
        fibermap : table

    Returns:
        QFrame object
    """
    log=get_logger()
    log.info("Starting...")

    t0=time.time()

    wavemin = xytraceset.wavemin
    wavemax = xytraceset.wavemax
    xcoef   = xytraceset.x_vs_wave_traceset._coeff
    ycoef   = xytraceset.y_vs_wave_traceset._coeff

    if fibers is None:
        if fibermap is not None:
            fibers = fibermap['FIBER']
        else:
            spectrograph = 0
            if "CAMERA" in image.meta :
                camera=image.meta["CAMERA"].strip()
                spectrograph = int(camera[-1])
                log.info("camera='{}' -> spectrograph={}. I AM USING THIS TO DEFINE THE FIBER NUMBER (ASSUMING 500 FIBERS PER SPECTRO).".format(camera,spectrograph))

            fibers = np.arange(xcoef.shape[0])+500*spectrograph

    #log.info("wavelength range : [%f,%f]"%(wavemin,wavemax))

    if image.mask is not None :
        image.ivar *= (image.mask==0)

    #  Applying a mask that keeps positive value to get the Variance by inversing the inverse variance.
    var=np.zeros(image.ivar.size)
    ok=image.ivar.ravel()>0
    var[ok] = 1./image.ivar.ravel()[ok]
    var=var.reshape(image.ivar.shape)

    badimage=(image.ivar==0)

    n0 = image.pix.shape[0]
    n1 = image.pix.shape[1]

    frame_flux = np.zeros((fibers.size,n0))
    frame_ivar = np.zeros((fibers.size,n0))
    frame_wave = np.zeros((fibers.size,n0))

    frame_sigma = None
    ysigcoef = None
    if save_sigma :
        if  xytraceset.ysig_vs_wave_traceset is None :
            log.warning("will not save sigma in qframe because missing in traceset")
        else :
            frame_sigma = np.zeros((fibers.size,n0))
            ysigcoef    = xytraceset.ysig_vs_wave_traceset._coeff

    xx         = np.tile(np.arange(n1),(n0,1))
    hw = width//2


    twave=np.linspace(wavemin, wavemax, n0//4) # this number of bins n0//p is calibrated to give a negligible difference of wavelength precision
    rwave=(twave-wavemin)/(wavemax-wavemin)*2-1.
    y=np.arange(n0).astype(float)


    dwave = np.zeros(n0)
    for f,fiber in enumerate(fibers) :
        log.debug("extracting fiber #%03d"%fiber)
        ty = legval(rwave, ycoef[f])
        tx = legval(rwave, xcoef[f])
        frame_wave[f] = np.interp(y,ty,twave)
        x_of_y        = np.interp(y,ty,tx)

        i=np.where(y<ty[0])[0]
        if i.size>0 : # need extrapolation
            frame_wave[f,i] = twave[0]+(twave[1]-twave[0])/(ty[1]-ty[0])*(y[i]-ty[0])
        i=np.where(y>ty[-1])[0]
        if i.size>0 : # need extrapolation
            frame_wave[f,i] = twave[-1]+(twave[-2]-twave[-1])/(ty[-2]-ty[-1])*(y[i]-ty[-1])

        dwave[1:]     = frame_wave[f,1:]-frame_wave[f,:-1]
        dwave[0]      = 2*dwave[1]-dwave[2]
        if np.any(dwave<=0) :
            log.error("neg. or null dwave")
            raise ValueError("neg. or null dwave")

        frame_flux[f],frame_ivar[f] = numba_extract(image.pix,var,x_of_y,hw)
        # flux density
        frame_flux[f] /= dwave
        frame_ivar[f] *= dwave**2

        if frame_sigma is not None :
            ts = legval(rwave, ysigcoef[f])
            frame_sigma[f] = np.interp(y,ty,ts)

    t1=time.time()
    log.info(" done {} fibers in {:3.1f} sec".format(len(fibers),t1-t0))

    if fibermap is None:
        log.warning("setting up a fibermap to save the FIBER identifiers")
        fibermap = empty_fibermap(fibers.size)
        fibermap["FIBER"] = fibers
    else :
        indices = np.arange(fibermap["FIBER"].size)[np.in1d(fibermap["FIBER"],fibers)]
        fibermap = fibermap[:][indices]

    return QFrame(frame_wave, frame_flux, frame_ivar, mask=None, sigma=frame_sigma , fibers=fibers, meta=image.meta, fibermap=fibermap)
