import time
import numpy as np
import scipy.ndimage

from desiutil.log import get_logger
from desispec.linalg import spline_fit
from desispec.quickproc.qframe import QFrame

def quick_sky_subtraction(qframe) :
    log=get_logger()
    t0=time.time()
    log.info("Starting...")
    
    twave=np.mean(qframe.wave,axis=0)
    tflux=np.zeros(qframe.flux.shape)
    tivar=np.zeros(qframe.flux.shape)

    if qframe.mask is not None :
        qframe.ivar *= (qframe.mask==0)

    for i in range(qframe.flux.shape[0]) :
        jj=(qframe.ivar[i]>0)
        tflux[i]=np.interp(twave,qframe.wave[i,jj],qframe.flux[i,jj])
    
    if qframe.fibermap is None :
        log.error("Empty fibermap in qframe, cannot know which are the sky fibers!")
        raise RuntimeError("Empty fibermap in qframe, cannot know which are the sky fibers!")
    
    skyfibers = np.where(qframe.fibermap["OBJTYPE"]=="SKY")[0]
    if skyfibers.size==0 :
       log.error("No sky fibers!")
       raise RuntimeError("No sky fibers!") 
    
    sky  = np.median(tflux[skyfibers],axis=0)
    for i in range(qframe.flux.shape[0]) :
        qframe.flux[i] -= np.interp(qframe.wave[i],twave,sky)
    
    t1=time.time()
    log.info(" done in {:3.1f} sec".format(t1-t0))

    #import matplotlib.pyplot as plt
    #plt.plot(twave,sky)
    #plt.show()
    
