"""
desispec.qproc.qfiberflat
=========================

"""
import time
import numpy as np
import scipy.ndimage

from desiutil.log import get_logger
from desispec.linalg import spline_fit
from desispec.qproc.qframe import QFrame
from desispec.fiberflat import FiberFlat

def qproc_apply_fiberflat(qframe,fiberflat,return_flat=False) :
    """
    Apply a fiber flat to a qframe.

    Inputs:
       qframe: desispec.qproc.qframe.QFrame object which will be modified
       fiberflat: desispec.fiberflat.FiberFlat object with the flat to apply

    Optional:
       return_flat : if True, returns the flat field that has been applied

    Returns nothing or the flat that has been applied.

    """
    log = get_logger()

    if return_flat :
        flat=np.ones(qframe.flux.shape)
    for j in range(qframe.flux.shape[0]) :
        k=j#np.where(fiberflat.fibers==qframe.fibers[j])[0]
        ii=np.where((fiberflat.fiberflat[k]!=0)&(fiberflat.ivar[k]>0)&(fiberflat.mask[k]==0))[0]
        if ii.size>0 :
            tmp=np.interp(qframe.wave[j],fiberflat.wave[ii],fiberflat.fiberflat[k,ii],left=0,right=0)
            qframe.flux[j] *= (tmp>0)/(tmp+(tmp==0))
            qframe.ivar[j] *= tmp**2
            if return_flat : flat[j]=tmp
        else :
            qframe.ivar[j] = 0.
    if return_flat :
        return flat

def qproc_compute_fiberflat(qframe,niter_meanspec=4,nsig_clipping=3.,spline_res_clipping=20.,spline_res_flat=5.) :
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
        if np.any(jj):
            tflux[i]=np.interp(twave,qframe.wave[i,jj],qframe.flux[i,jj])
            tivar[i]=np.interp(twave,qframe.wave[i,jj],qframe.ivar[i,jj],left=0,right=0)

    # iterative loop to a absorb constant term in fiber
    if 1 : # simple scaling per fiber
        a=np.ones(tflux.shape[0])
        for iter in range(niter_meanspec) :
            mflux=np.median(a[:,np.newaxis]*tflux,axis=0)
            for i in range(qframe.flux.shape[0]) :
                a[i] = np.median(tflux[i,mflux>0]/mflux[mflux>0])

    else : # polynomial fit does not improve much and s more fragile
        x=np.linspace(-1,1,tflux.shape[1])
        pol=np.ones(tflux.shape)
        for iteration in range(niter_meanspec) :
            if iteration>0 :
                for i in range(tflux.shape[0]) :
                    jj=(mflux>0)&(tivar[i]>0)
                    c = np.polyfit(x[jj], tflux[i, jj] / mflux[jj], 1,
                                   w=mflux[jj] * np.sqrt(tivar[i, jj]))
                    pol[i] = np.poly1d(c)(x)
            mflux=np.median(pol*tflux,axis=0)

    # trivial fiberflat
    fflat=tflux/(mflux+(mflux==0))
    fivar=tivar*mflux**2

    mask=np.zeros((fflat.shape), dtype='uint32')
    chi2=0
    # special case with test slit
    mask_lines = ( qframe.flux.shape[0]<50 )
    if mask_lines :
        log.warning("Will interpolate over absorption lines in input continuum spectrum from illumination bench")


    # spline fit to reject outliers and smooth the flat
    for fiber in range(fflat.shape[0]) :
        # check for completely masked fiber
        if np.all(fivar[fiber] == 0.0):
            log.warning(f'All wavelengths of fiber {fiber} are masked; setting fflat=1 fivar=0')
            fflat[fiber] = 1.0
            fivar[fiber] = 0.0
            mask[fiber] = 1
            continue

        # iterative spline fit
        max_rej_it=5# not more than 5 pixels at a time
        max_bad=1000
        nbad_tot=0
        for loop in range(20) :
            good=(fivar[fiber]>0)
            splineflat = spline_fit(twave,twave[good],fflat[fiber,good],required_resolution=spline_res_clipping,input_ivar=fivar[fiber,good],max_resolution=3*spline_res_clipping)
            fchi2 = fivar[fiber]*(fflat[fiber]-splineflat)**2
            bad=np.where(fchi2>nsig_clipping**2)[0]
            if bad.size>0 :
                if bad.size>max_rej_it : # not more than 5 pixels at a time
                    ii=np.argsort(fchi2[bad])
                    bad=bad[ii[-max_rej_it:]]
                fivar[fiber,bad] = 0
                nbad_tot += len(bad)
                #log.warning("iteration {} rejecting {} pixels (tot={}) from fiber {}".format(loop,len(bad),nbad_tot,fiber))
                if nbad_tot>=max_bad:
                    fivar[fiber,:]=0
                    log.warning("1st pass: rejecting fiber {} due to too many (new) bad pixels".format(fiber))
            else :
                break

        chi2 += np.sum(fchi2)

        min_ivar = 0.1*np.median(fivar[fiber])
        med_flat = np.median(fflat[fiber])

        good=(fivar[fiber]>0)
        splineflat = spline_fit(twave,twave[good],fflat[fiber,good],required_resolution=spline_res_flat,input_ivar=fivar[fiber,good],max_resolution=3*spline_res_flat)
        fflat[fiber] = splineflat # replace by spline

        ii=np.where(fivar[fiber]>min_ivar)[0]
        if ii.size<2 :
            fflat[fiber] = 1
            fivar[fiber] = 0

        # set flat in unknown edges to median value of fiber (and ivar to 0)
        b=ii[0]
        e=ii[-1]+1
        fflat[fiber,:b]=med_flat # default
        fivar[fiber,:b]=0
        mask[fiber,:b]=1 # need to change this
        fflat[fiber,e:]=med_flat # default
        fivar[fiber,e:]=0
        mask[fiber,e:]=1 # need to change this

        # internal interpolation
        bad=(fivar[fiber][b:e]<=min_ivar)
        good=(fivar[fiber][b:e]>min_ivar)
        fflat[fiber][b:e][bad]=np.interp(twave[b:e][bad],twave[b:e][good],fflat[fiber][b:e][good])

        # special case with test slit
        if mask_lines :
            if qframe.meta["camera"].upper()[0] == "B" :
                jj=((twave>3900)&(twave<3960))|((twave>4350)&(twave<4440))|(twave>5800)
            elif qframe.meta["camera"].upper()[0] == "R" :
                jj=(twave<5750)
            else :
                jj=(twave<7550)|(twave>9800)
            if np.sum(jj)>0 :
                njj=np.logical_not(jj)
                fflat[fiber,jj] = np.interp(twave[jj],twave[njj],fflat[fiber,njj])

    ndata=np.sum(fivar>0)
    if ndata>0 :
        chi2pdf = chi2/ndata
    else :
        chi2pdf = 0

    t1=time.time()
    log.info(" done in {:3.1f} sec".format(t1-t0))

    # return a fiberflat object ...

    return FiberFlat(twave, fflat, fivar, mask, mflux,chi2pdf=chi2pdf)
    #TO FINISH
