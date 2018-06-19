import time
import numpy as np
import scipy.ndimage

from desiutil.log import get_logger
from .qframe import QFrame

def quick_fiberflat(qframe,nloop=4,min_median_flux=100.,max_flat_uncertainty=0.1,median_filter_width=50) :    
    """
    Fast estimation of fiberflat
    """
    
    log = get_logger()
    
    t0=time.time()
    log.info("Starting...")
    twave=np.mean(qframe.wave,axis=0)
    tflux=np.zeros(qframe.flux.shape)
    tivar=np.zeros(qframe.flux.shape)
    
    for i in range(qframe.flux.shape[0]) :
        tflux[i]=np.interp(twave,qframe.wave[i],qframe.flux[i])
   
    # iterative loop to absorb constant term in fiber (should have more parameters)
    
    for loop in range(nloop) :
        mflux=np.median(tflux,axis=0)
        for i in range(qframe.flux.shape[0]) :
            a = np.median(tflux[i,mflux>0]/mflux[mflux>0])
            if a>0 : tflux[i] /= a
    
    
    # go back to original grid
    ok=(mflux>min_median_flux)
    for i in range(qframe.flux.shape[0]) :
        tmp = np.interp(qframe.wave[i],twave[ok],mflux[ok],left=0,right=0)
        good=(tmp!=0)
        tflux[i,good] = qframe.flux[i,good]/tmp[good]
        tivar[i,good] = qframe.ivar[i,good]*tmp[good]**2
        good &= (qframe.ivar[i]>0)
        if qframe.mask is not None : 
            good &= (qframe.mask[i]==0)
        bad  = np.logical_not(good)
        tflux[i,bad] = np.interp(qframe.wave[i,bad],qframe.wave[i,good],tflux[i,good])
        
        # sliding median to detect cosmics
        tmp  = scipy.ndimage.filters.median_filter(tflux[i],median_filter_width,mode='constant')
        diff = np.abs(tflux[i]-tmp)
        mdiff = scipy.ndimage.filters.median_filter(diff,200,mode='constant')
        good = diff<3*mdiff
        bad  = np.logical_not(good)
        tflux[i,bad] = np.interp(qframe.wave[i,bad],qframe.wave[i,good],tflux[i,good])
    
    t1=time.time()
    log.info(" done in {:3.1f} sec".format(t1-t0))
    
    bad=(tivar<1./max_flat_uncertainty**2)
    tivar[bad]=0.
    tflux[bad]=1. # default (for nice plots)
    

    # now smooth the flat
    import matplotlib.pyplot as plt
    for i in range(qframe.flux.shape[0]) :
        plt.plot(twave,tflux[i])
    
    plt.show()
    #t2=time.time(); log.info(" done in {} sec".format(t2-t1)) ; t1=t2

