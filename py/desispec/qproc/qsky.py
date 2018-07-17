import time
import numpy as np
import scipy.ndimage

from desiutil.log import get_logger
from desispec.qproc.qframe import QFrame

def qproc_sky_subtraction(qframe,return_skymodel=False) :
    log=get_logger()
    t0=time.time()
    log.info("Starting...")
    
    twave=np.linspace(np.min(qframe.wave),np.max(qframe.wave),qframe.wave.shape[1]*2) # oversampling
    tflux=np.zeros((qframe.flux.shape[0],twave.size))
    tivar=np.zeros((qframe.flux.shape[0],twave.size))
    
    if return_skymodel :
        sky=np.zeros(qframe.flux.shape)
    
    if qframe.mask is not None :
        qframe.ivar *= (qframe.mask==0)
        
    if qframe.fibermap is None :
        log.error("Empty fibermap in qframe, cannot know which are the sky fibers!")
        raise RuntimeError("Empty fibermap in qframe, cannot know which are the sky fibers!")
    
    skyfibers = np.where(qframe.fibermap["OBJTYPE"]=="SKY")[0]
    if skyfibers.size==0 :
       log.error("No sky fibers!")
       raise RuntimeError("No sky fibers!") 
    log.info("Sky fibers: {}".format(skyfibers))
    
    for loop in range(5) : # I need several iterations to remove the effect of the wavelength solution noise
        
        for i in skyfibers :
            jj=(qframe.ivar[i]>0)
            tflux[i]=np.interp(twave,qframe.wave[i,jj],qframe.flux[i,jj])
        
        ttsky  = np.median(tflux[skyfibers],axis=0)
        
        if return_skymodel :
            for i in range(qframe.flux.shape[0]) :
                tmp=np.interp(qframe.wave[i],twave,ttsky)
                jj=(qframe.flux[i]!=0)
                qframe.flux[i,jj] -= tmp[jj]
                sky[i] += tmp
        else :
            for i in range(qframe.flux.shape[0]) :
                jj=(qframe.flux[i]!=0)
                qframe.flux[i,jj] -= np.interp(qframe.wave[i,jj],twave,ttsky)
        

    t1=time.time()
    log.info(" done in {:3.1f} sec".format(t1-t0))
    
    if return_skymodel :
        return sky
