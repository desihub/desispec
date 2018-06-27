"""
desispec.cosmics
================

Utility functions to find cosmic rays
"""


from desiutil.log import get_logger
import numpy as np
import math
import copy
import scipy.ndimage
import time

import numba

from desispec.maskbits import ccdmask
from desispec.maskbits import specmask



def reject_cosmic_rays_1d(frame,nsig=3,psferr=0.05) :
    """Use resolution matrix in frame to detect spikes in the spectra that
    are narrower than the PSF, and mask them"""
    log=get_logger()
    
    log.info("subtract continuum to flux")
    tflux=np.zeros(frame.flux.shape)
    for fiber in range(frame.nspec) :
        tflux[fiber]=frame.flux[fiber]-scipy.ndimage.filters.median_filter(frame.flux[fiber],200,mode='constant')
    log.info("done")
        
    # we do not use the mask here because we want to re-detect cosmics
    # to broaden the masked area if needed
    # variance of flux
    var=(frame.ivar>0)/(frame.ivar+(frame.ivar==0)) 
    var[var==0]=(np.max(var)*1000.) # a large number
    var += (psferr*tflux)**2 # add a psf error
    
    # positive peaks
    peaks=np.zeros(tflux.shape)
    peaks[:,1:-1]=(tflux[:,1:-1]>tflux[:,:-2])*(tflux[:,1:-1]>tflux[:,2:])
    
    # gradients on both sides
    dfp=np.zeros(tflux.shape)
    dfm=np.zeros(tflux.shape)    
    dfp[:,1:-1]=(tflux[:,1:-1]-tflux[:,2:])
    dfm[:,1:-1]=(tflux[:,1:-1]-tflux[:,:-2])
    # variance of gradients
    vp=np.zeros(tflux.shape)
    vm=np.zeros(tflux.shape)
    vp[:,1:-1]=(var[:,1:-1]+var[:,2:])
    vm[:,1:-1]=(var[:,1:-1]+var[:,:-2])
    # chi2 of gradients
    chi2p=dfp**2*(dfp>0)*(tflux>0)/(vp+(vp==0))
    chi2m=dfm**2*(dfm>0)*(tflux>0)/(vm+(vm==0))
        
    for fiber in range(chi2m.shape[0]) :
        R=frame.R[fiber]
        
        # potential cosmics 
        selection=np.where( ( (chi2p[fiber]>nsig**2) | (chi2m[fiber]>nsig**2) ) & (peaks[fiber]>0) )[0]
        if selection.size==0 : continue # no potential cosmic
        
        # loop on peaks
        for i in selection :
            
            # relative variations
            rdfpi=dfp[fiber,i]/tflux[fiber,i]
            rdfmi=dfm[fiber,i]/tflux[fiber,i]
            # error
            errp=np.sqrt(vp[fiber,i])/tflux[fiber,i]
            errm=np.sqrt(vm[fiber,i])/tflux[fiber,i]
            
            # profile from resolution matrix
            r  =  R.data[:,i]
            d  = r.size//2
            rdrp = 1-r[d+1]/r[d]
            rdrm = 1-r[d-1]/r[d]
            snrp = (rdfpi-rdrp)/errp
            snrm = (rdfmi-rdrm)/errm
            
            # S/N at peak (difference between peak at i and PSF profile at peak from adjacent pixels) 
            # snr  = (dflux[fiber,i]-psfpeak)/np.sqrt( 1./(divar[fiber,i]+(divar[fiber,i]==0))+dr[d]**2/a )
            snr=max(snrp,snrm)
            if snr>nsig :
                # also mask neighboring pixels if >nsig
                d=2
                b=i-d
                e=i+d+1
                previous_nmasked=np.sum(frame.mask[fiber,b:e]>0)
                frame.mask[fiber,b:e][np.sqrt(frame.ivar[fiber,b:e])*tflux[fiber,b:e]>nsig] |= specmask.COSMIC
                new_nmasked=np.sum(frame.mask[fiber,b:e]>0)
                nmasked=(new_nmasked-previous_nmasked)
                if nmasked>0 :
                    if previous_nmasked>0 :
                        log.info("fiber {} wave={} S/N={} broaden cosmic mask by {} pix".format(fiber,int(frame.wave[i]),int(snr),nmasked))
                    else :
                        log.info("fiber {} wave={} S/N={} add cosmic mask of {} pix".format(fiber,int(frame.wave[i]),int(snr),nmasked))
    log.info("done")

@numba.jit
def dilate_numba(input_boolean_array,include_input=False) :
    output_boolean_array = np.zeros(input_boolean_array.shape,dtype=bool)
    if include_input :
        output_boolean_array |= input_boolean_array
    for i0 in range(1,input_boolean_array.shape[0]-1) :
        for i1 in range(1,input_boolean_array.shape[1]-1) :
            if input_boolean_array[i0,i1] :
                output_boolean_array[i0-1,i1]=True
                output_boolean_array[i0+1,i1]=True
                output_boolean_array[i0,i1-1]=True
                output_boolean_array[i0,i1+1]=True
                output_boolean_array[i0-1,i1-1]=True
                output_boolean_array[i0+1,i1+1]=True
                output_boolean_array[i0-1,i1+1]=True
                output_boolean_array[i0+1,i1-1]=True
    return output_boolean_array



@numba.jit
def reject_cosmic_rays_ala_sdss_numba(pix,ivar,selection,psf_gradients,nsig,cfudge,c2fudge) :
    """Cosmic ray rejection following the implementation in SDSS/BOSS.
    (see idlutils/src/image/reject_cr_psf.c and idlutils/pro/image/reject_cr.pro)

    This routine is a single call, similar to IDL routine reject_cr_single
    Called by reject_cosmic_rays_ala_sdss with an iteration

    Input is a pre-processed image : desispec.Image
    Ouput is a rejection mask of the same size as the image

    Args:
       pix: input desispec.Image.pix (counts in pixels, 2D image)
       ivar: inverse variance of pix
       selection: array of booleans
       psf_gradients: 1D array of size 4, for 4 axes: horizontal,vertical and 2 diagonals
       nsig: number of sigma above background required
       cfudge: number of sigma inconsistent with PSF required
       c2fudge:  fudge factor applied to PSF
    """
    
    
    n0    = pix.shape[0]
    n1    = pix.shape[1]
    
    # definition of axis
    naxis = psf_gradients.size
    dd = np.zeros((naxis,2),dtype=int)
    for a in range(naxis) :
        if a==0 : 
            dd[a,0]=0
            dd[a,1]=1
        elif a==1 :
            dd[a,0]=1
            dd[a,1]=0
        elif a==2 :
            dd[a,0]=1
            dd[a,1]=1
        else :
            dd[a,0]=1
            dd[a,1]=-1
        
    rejection=np.zeros(pix.shape,dtype=bool)
    
    for i0 in range(1,n0-1) :
        for i1 in range(1,n1-1) :
            
            if (not selection[i0,i1]) or ivar[i0,i1]<=0 : continue
            
            # first criterion, signal in pix must be significantly higher than neighbours (=back)
            # in all directions
            # JG comment : this does not look great for muon tracks that are perfectly aligned
            # with one the axis.
            # I change the algorithm to accept 3 out of 4 valid tests
            first_criterion=1 # why do I start at 1 ?????
            
            # second criterion, rejected if at least for one axis
            # the background values "back" are not consistent with PSF given the central pixel value
            # here the number of sigmas is the parameter cfudge
            # c2fudge alters the PSF
            second_criterion=False
            
            pixii=pix[i0,i1]
            sigii=1/np.sqrt(ivar[i0,i1])
            
            for a in range(naxis) :
                d0=dd[a,0]
                d1=dd[a,1]
                
                ivar0=ivar[i0-d0,i1-d1]
                ivar1=ivar[i0+d0,i1+d1]
                nvalid=int(ivar0>0)+int(ivar1>0)
                nvalidinv=float(nvalid>0)/(nvalid+(nvalid==0))
                back=((ivar0>0)*pix[i0-d0,i1-d1]+(ivar1>0)*pix[i0+d0,i1+d1])*nvalidinv
                sigmaback=np.sqrt((ivar0>0)/(ivar0+(ivar0==0))+(ivar1>0)/(ivar1+(ivar1==0)))*nvalidinv
                
                first_criterion += (pixii>(back+nsig*sigii))
                second_criterion |= (((pixii-cfudge*sigii)*c2fudge*psf_gradients[a]) > ( back+cfudge*sigmaback ))
            
            rejection[i0,i1] = ( (first_criterion>=3) & second_criterion )
            
    return rejection
    
def reject_cosmic_rays_ala_sdss_single(pix,ivar,selection,psf_gradients,nsig,cfudge,c2fudge) :
    """Cosmic ray rejection following the implementation in SDSS/BOSS.
    (see idlutils/src/image/reject_cr_psf.c and idlutils/pro/image/reject_cr.pro)

    This routine is a single call, similar to IDL routine reject_cr_single
    Called by reject_cosmic_rays_ala_sdss with an iteration

    Input is a pre-processed image : desispec.Image
    Ouput is a rejection mask of the same size as the image

    Args:
       pix: input desispec.Image.pix (counts in pixels, 2D image)
       ivar: inverse variance of pix
       selection: array of booleans
       psf_gradients: 1D array of size 4, for 4 axes: horizontal,vertical and 2 diagonals
       nsig: number of sigma above background required
       cfudge: number of sigma inconsistent with PSF required
       c2fudge:  fudge factor applied to PSF
    """
    log=get_logger()
    log.debug("starting with nsig=%2.1f cfudge=%2.1f c2fudge=%2.1f"%(nsig,cfudge,c2fudge))

    # psf is precomputed for each camera
    # using psf_for_cosmics.py --psffile psf-{b,r,z}0-00000000.fits
    # today based on data challenge 2 results , psf files from
    # /project/projectdirs/desi/spectro/redux/alpha-3/calib2d/psf/20150107
    #
    naxis=psf_gradients.size
    
    
    # pixel values and inverse variance 
    # is flatted to ease the data manipulation
    # we only consider data 1 pixel off from CCD edge because neighboring
    # pixels are used
    
    tselection=selection[1:-1,1:-1]
    
    tpix=pix[1:-1,1:-1][tselection].ravel()
    tpixivar=ivar[1:-1,1:-1][tselection].ravel()
    
    
    # there are 4 axis (horizontal,vertical,and 2 diagonals)
    # for each axis, there is a pair of two pixels on each side of the pixel of interest
    # pairpix is the pixel values
    # pairivar their inverse variance (accounting for pre-existing mask)
    naxis=4
    pairpix=np.zeros((naxis,2,tpix.size))
    pairpix[0,0]=pix[1:-1,2:][tselection].ravel()
    pairpix[0,1]=pix[1:-1,:-2][tselection].ravel()
    pairpix[1,0]=pix[2:,1:-1][tselection].ravel()
    pairpix[1,1]=pix[:-2,1:-1][tselection].ravel()
    pairpix[2,0]=pix[2:,2:][tselection].ravel()
    pairpix[2,1]=pix[:-2,:-2][tselection].ravel()
    pairpix[3,0]=pix[2:,:-2][tselection].ravel()
    pairpix[3,1]=pix[:-2,2:][tselection].ravel()
    pairivar=np.zeros((naxis,2,tpix.size))
    pairivar[0,0]=ivar[1:-1,2:][tselection].ravel()
    pairivar[0,1]=ivar[1:-1,:-2][tselection].ravel()
    pairivar[1,0]=ivar[2:,1:-1][tselection].ravel()
    pairivar[1,1]=ivar[:-2,1:-1][tselection].ravel()
    pairivar[2,0]=ivar[2:,2:][tselection].ravel()
    pairivar[2,1]=ivar[:-2,:-2][tselection].ravel()
    pairivar[3,0]=ivar[2:,:-2][tselection].ravel()
    pairivar[3,1]=ivar[:-2,2:][tselection].ravel()

    # back and sigmaback are the average values of each pair and their error
    # (same notation as SDSS code)
    tmp=np.sum(pairivar>0,axis=1).astype(float)
    tmp=(tmp>0)/(tmp+(tmp==0))
    back=np.sum(pairpix*(pairivar>0),axis=1)*tmp
    sigmaback=np.sqrt(np.sum((pairivar>0)/(pairivar+(pairivar==0)),axis=1))*tmp

    log.debug("mean pix = %f"%np.mean(tpix))
    log.debug("mean back = %f"%np.mean(back))
    log.debug("mean sigmaback = %f"%np.mean(sigmaback))
    
    # first criterion, signal in pix must be significantly higher than neighbours (=back)
    # in all directions
    # JG comment : this does not look great for muon tracks that are perfectly aligned
    # with one the axis.
    # I change the algorithm to accept 3 out of 4 valid tests
    first_criterion=np.ones(tpix.shape)
    nonullivar=tpixivar>0
    tmp=np.zeros(tpix.shape)
    tmp[nonullivar]=tpix[nonullivar]-nsig/np.sqrt(tpixivar[nonullivar])
    for a in range(naxis) :
        first_criterion += (tmp>back[a])
    first_criterion=(first_criterion>=3).astype(bool)

    # second criterion, rejected if at least for one axis
    # the values back are not consistent with PSF given the central
    # pixel value
    # here the number of sigmas is the parameter cfudge
    # c2fudge alters the PSF
    second_criterion=np.zeros(tpix.shape).astype(bool)
    tmp=np.zeros(tpix.shape)
    tmp[nonullivar]=tpix[nonullivar]-cfudge/np.sqrt(tpixivar[nonullivar])
    for a in range(naxis) :
        second_criterion |= ( tmp*c2fudge*psf_gradients[a] > ( back[a]+cfudge*sigmaback[a] ) )


    log.debug("npix selected                       = %d"%tpix.size)
    log.debug("npix rejected 1st criterion         = %d"%np.sum(first_criterion))
    log.debug("npix rejected 1st and 2nd criterion = %d"%np.sum(first_criterion&second_criterion))
    
    # remap to original shape
    rejection=np.zeros(pix.shape).astype(bool)
    rejection[1:-1,1:-1][tselection] = (first_criterion&second_criterion).reshape(pix[1:-1,1:-1][tselection].shape)
    return rejection

def reject_cosmic_rays_ala_sdss(img,nsig=6.,cfudge=3.,c2fudge=0.8,niter=6,dilate=True) :
    """Cosmic ray rejection following the implementation in SDSS/BOSS.
    (see idlutils/src/image/reject_cr_psf.c and idlutils/pro/image/reject_cr.pro)

    This routine is calling several times reject_cosmic_rays_ala_sdss_single,
    similar to IDL routine reject_cr.
    There is an optionnal dilatation of the mask by one pixel, as done in sdssproc.pro for SDSS

    Input is a pre-processed image : desispec.Image
    Ouput is a rejection mask of the same size as the image

    Args:
       img: input desispec.Image
       nsig: number of sigma above background required
       cfudge: number of sigma inconsistent with PSF required
       c2fudge:  fudge factor applied to PSF
       niter: number of iterations on neighboring pixels of rejected pixels
       dilate: force +1 pixel dilation of rejection mask
    """
    log=get_logger()
    log.info("starting with nsig=%2.1f cfudge=%2.1f c2fudge=%2.1f"%(nsig,cfudge,c2fudge))
    t0=time.time()

    tivar=img.ivar*(img.mask==0)*(img.ivar>0)
    
    
    band=img.camera[0].lower()
    if band == 'b':
        psf_gradients=np.array([0.366247,0.391422,0.172965,0.184552])
    elif band == 'r':
        psf_gradients=np.array([0.39508155,0.2951822,0.13044542,0.14904523])
    elif band == 'z':
        psf_gradients=np.array([0.513852,0.537679,0.297071,0.276298])
    else :
        log.error("do not have psf info for band='%s'"%band)
        raise KeyError("do not have psf info for band='%s'"%band)
    
    selection = ((img.pix*np.sqrt(tivar))>nsig)
    
    use_numba = True
    
    if use_numba :
        rejected   = reject_cosmic_rays_ala_sdss_numba(img.pix,tivar,selection,psf_gradients,nsig=nsig,cfudge=cfudge,c2fudge=c2fudge)
    else :
        rejected  = reject_cosmic_rays_ala_sdss_single(img.pix,tivar,selection,psf_gradients,nsig=nsig,cfudge=cfudge,c2fudge=c2fudge)
    
    
    log.info("first pass: %d pixels rejected"%(np.sum(rejected)))
    
    if niter > 0 :        
        
        for iteration in range(niter) :

            # if np.sum(rejected)==0 : break
            if use_numba :
                neighbors = dilate_numba(rejected,False)
            else :
                neighbors = np.zeros(rejected.shape,dtype=bool)
                # left and right neighbors
                neighbors[1:,:]  |= rejected[:-1,:]
                neighbors[:-1,:] |= rejected[1:,:]
                neighbors[:,1:]  |= rejected[:,:-1]
                neighbors[:,:-1] |= rejected[:,1:]
                # adding diagonals (not in original SDSS version)
                neighbors[1:,1:]  |= rejected[:-1,:-1]
                neighbors[:-1,:-1]  |= rejected[1:,1:]
                neighbors[1:,:-1]  |= rejected[:-1,1:]
                neighbors[:-1,1:]  |= rejected[1:,:-1]
            neighbors &= (rejected==False) # excluded already rejected pixel
            tivar[rejected] = 0. # mask already rejected pixels for the calculation of the background of the neighbors

            # rerun with much more strict cuts
            if use_numba :
                newrejected=reject_cosmic_rays_ala_sdss_numba(img.pix,tivar,neighbors,psf_gradients,nsig=3.,cfudge=0.,c2fudge=1.)
            else :
                newrejected=reject_cosmic_rays_ala_sdss_single(img.pix,tivar,neighbors,psf_gradients,nsig=3.,cfudge=0.,c2fudge=1.)
                
            log.info("at iter %d: %d new pixels rejected"%(iteration,np.sum(newrejected)))
            if np.sum(newrejected)<1 :            
                break
            rejected |= newrejected
        

    if dilate :
        log.debug("dilating cosmic ray mask")
        # now apply the dilatation included in sdssproc.pro
        # in IDL it is crmask = dilate(crmask, replicate(1,3,3))
        if use_numba :
            rejected = dilate_numba(rejected,True)
        else :
            tmp=rejected.copy()
            rejected[1:,:]  |= tmp[:-1,:]
            rejected[:-1,:] |= tmp[1:,:]
            rejected[:,1:]  |= tmp[:,:-1]
            rejected[:,:-1] |= tmp[:,1:]
            rejected[1:,1:]  |= tmp[:-1,:-1]
            rejected[:-1,:-1]  |= tmp[1:,1:]
            rejected[1:,:-1]  |= tmp[:-1,1:]
            rejected[:-1,1:]  |= tmp[1:,:-1]
        
    t1=time.time()
    log.info("end : {} pixels rejected in {:3.1f} sec".format(np.sum(rejected),t1-t0))
    return rejected

def reject_cosmic_rays(img,nsig=5.,cfudge=3.,c2fudge=0.9,niter=30,dilate=True) :
    """Cosmic ray rejection
    Input is a pre-processed image : desispec.Image
    The image mask is modified

    Args:
       img: input desispec.Image

    """
    rejected=reject_cosmic_rays_ala_sdss(img,nsig=nsig,cfudge=cfudge,c2fudge=c2fudge,niter=niter,dilate=dilate)
    img.mask[rejected] |= ccdmask.COSMIC
