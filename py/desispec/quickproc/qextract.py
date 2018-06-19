import time
import numpy as np
from numpy.polynomial.legendre import legval
import numba

from desiutil.log import get_logger
from ..xytraceset import XYTraceSet
from ..image import Image
#from specter.util import legval_numba
from .qframe import QFrame


@numba.jit
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



def boxcar_extraction(xytraceset, image, fibers=None, width=7) :    
    """
    Fast boxcar extraction of spectra from a preprocessed image and a trace set
    
    Args:
        xytraceset : DESI XYTraceSet object
        image : DESI preprocessed Image object

    Optional:   
        fibers : 1D np.array of int (default is all fibers, the first fiber is always = 0)
        width  : extraction boxcar width, default is 7

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
    
    if fibers is None :
        fibers = np.arange(xcoef.shape[0])
    
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
    xx         = np.tile(np.arange(n1),(n0,1))
    hw = width//2
    
    
    twave=np.linspace(wavemin, wavemax, n0//4) # this number of bins n0//p is calibrated to give a negligible difference of wavelength precision
    rwave=(twave-wavemin)/(wavemax-wavemin)*2-1.
    y=np.arange(n0).astype(float)
    
    
    
    for f,fiber in enumerate(fibers) :
        log.debug("extracting fiber #%03d"%fiber)
        ty = legval(rwave, ycoef[fiber])
        tx = legval(rwave, xcoef[fiber])
        frame_wave[f] = np.interp(y,ty,twave)
        x_of_y        = np.interp(y,ty,tx)        
        frame_flux[f],frame_ivar[f] = numba_extract(image.pix,var,x_of_y,hw)
        
    t1=time.time()
    log.info(" done {} fibers in {:3.1f} sec".format(len(fibers),t1-t0))
    
    return QFrame(frame_wave, frame_flux, frame_ivar, mask=None, fibers=fibers)
