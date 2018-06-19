import time
import numpy as np
import scipy.ndimage

from desiutil.log import get_logger
from desispec.linalg import spline_fit
from desispec.quickproc.qframe import QFrame

def quick_apply_fiberflat(qframe,qflat) :

    log = get_logger()

    if len(qframe.fibers) == len(qflat.fibers) and np.sum(qframe.fibers != qflat.fibers)==0  :
        ii=(qflat.flux!=0)
        qframe.flux[ii] /= qflat.flux[ii]
        qframe.ivar[ii] *= qflat.flux[ii]**2
        qframe.ivar[(qflat.flux<=0)|(qflat.ivar==0)] = 0.
    else :
        for j in range(qframe.flux.shape[0]) :
            k=np.where(qflat.fibers==qframe.fibers[j])[0]
            if k.size != 1 :
                log.Error("No fiber {} in flat".format(qframe.fibers[j]))
                raise ValueError("No fiber {} in flat".format(qframe.fibers[j]))
            k=k[0]
            ii=(qflat.flux[k]!=0)
            qframe.flux[j,ii] /= qflat.flux[k,ii]
            qframe.ivar[j,ii] *= qflat.flux[k,ii]**2
            qframe.ivar[j,(qflat.flux[k]<=0)|(qflat.ivar[k]==0)] = 0.

def quick_compute_fiberflat(qframe,niter_meanspec=4,nsig_clipping=3.5,max_flat_uncertainty=0.1,spline_res=10.) :    
    """
    Fast estimation of fiberflat
    """
    
    log = get_logger()
    
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
   
    # iterative loop to absorb constant term in fiber (should have more parameters)
    
    for iter in range(niter_meanspec) :
        mflux=np.median(tflux,axis=0)
        for i in range(qframe.flux.shape[0]) :
            a = np.median(tflux[i,mflux>0]/mflux[mflux>0])
            if a>0 : tflux[i] /= a
    
    
    # go back to original grid
    ok=(mflux>100.)#np.median(mflux)*0.01)
    lowflux=np.where(np.logical_not(ok))[0]
    
    for fiber in range(qframe.flux.shape[0]) :
        
        tmp = np.interp(qframe.wave[fiber],twave[ok],mflux[ok],left=0,right=0)
        good=(tmp!=0)
        tflux[fiber,good] = qframe.flux[fiber,good]/tmp[good]
        tivar[fiber,good] = qframe.ivar[fiber,good]*tmp[good]**2
        good &= (qframe.ivar[fiber]>0)
        if qframe.mask is not None : 
            good &= (qframe.mask[fiber]==0)
        bad  = np.logical_not(good)
        tflux[fiber,bad] = np.interp(qframe.wave[fiber,bad],qframe.wave[fiber,good],tflux[fiber,good])
        tivar[fiber,bad] = 0.
        
        # iterative spline fit
        max_rej_it=5# not more than 5 pixels at a time
        max_bad=1000
        nbad_tot=0
        
        for loop in range(10) :
            
            good &= (tivar[fiber]>0)
            
            # add constrain at first and last wave
            #tflux[fiber,0]=tflux[fiber,-1]=1.
            #tivar[fiber,0]=tivar[fiber,-1]=1000.
            #good[0]=good[-1]=True
            
            splineflat = spline_fit(qframe.wave[fiber],qframe.wave[fiber,good],tflux[fiber,good],required_resolution=spline_res,input_ivar=tivar[fiber,good],max_resolution=3*spline_res)
            chi2 = tivar[fiber]*(tflux[fiber]-splineflat)**2
            bad=np.where(chi2>nsig_clipping**2)[0]
            if bad.size>0 :
                if bad.size>max_rej_it : # not more than 5 pixels at a time
                    ii=np.argsort(chi2[bad])
                    bad=bad[ii[-max_rej_it:]]
                tivar[fiber,bad] = 0
                nbad_tot += len(bad)
                log.warning("iteration {} rejecting {} pixels (tot={}) from fiber {}".format(loop,len(bad),nbad_tot,fiber))
                if nbad_tot>=max_bad:
                    tivar[fiber,:]=0
                    log.warning("1st pass: rejecting fiber {} due to too many (new) bad pixels".format(fiber))
            else :
                break
        tflux[fiber] = splineflat
        valid=np.where((tivar[fiber]>1./max_flat_uncertainty**2))[0]
        tflux[fiber,:valid[0]]    = 1.
        tflux[fiber,valid[-1]+1:] = 1.
        tivar[fiber,tivar[fiber]<1./max_flat_uncertainty**2]=0.
        
    t1=time.time()
    log.info(" done in {:3.1f} sec".format(t1-t0))
    
    return QFrame(qframe.wave, tflux, tivar, mask=None, fibers=qframe.fibers)

