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

def reject_cosmic_rays_ala_sdss_single(img,selection,nsig,cfudge,c2fudge) :
    """Cosmic ray rejection following the implementation in SDSS/BOSS.
    (see idlutils/src/image/reject_cr_psf.c and idlutils/pro/image/reject_cr.pro)

    This routine is a single call, similar to IDL routine reject_cr_single
    Called by reject_cosmic_rays_ala_sdss with an iteration

    Input is a pre-processed image : desispec.Image
    Ouput is a rejection mask of the same size as the image

    Args:
       img: input desispec.Image
       selection: selection of pixels to be tested (boolean array of same shape as img.pix[1:-1,1:-1])
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
    naxis=4


    band=img.camera[0].lower()

    if band == 'b':
        psf=np.array([0.366247,0.391422,0.172965,0.184552])
    elif band == 'r':
        psf=np.array([0.39508155,0.2951822,0.13044542,0.14904523])
    elif band == 'z':
        psf=np.array([0.513852,0.537679,0.297071,0.276298])
    else :
        log.error("do not have psf for camera '%s'"%img.camera)
        raise KeyError('No PSF for camera {}'.format(img.camera))

    # selection is now an argument (for neighbors)
    # selection=((img.pix*np.sqrt(img.ivar)*(img.mask==0))[1:-1,1:-1]>nsig).astype(bool)

    if np.sum(selection) ==0 :
        log.warning("no valid pixel above %2.1f sigma"%nsig)
        return np.zeros(img.pix.shape).astype(bool)

    # pixel values and inverse variance in selection (accounting for pre-existing mask)
    # it is flatted to ease the data manipulation
    # we only consider data 1 pixel off from CCD edge because neighboring
    # pixels are used
    pix=img.pix[1:-1,1:-1][selection].ravel()
    pixivar=(img.ivar[1:-1,1:-1][selection]*(img.mask[1:-1,1:-1][selection]==0)).ravel()

    # there are 4 axis (horizontal,vertical,and 2 diagonals)
    # for each axis, there is a pair of two pixels on each side of the pixel of interest
    # pairpix is the pixel values
    # pairivar their inverse variance (accounting for pre-existing mask)
    naxis=4
    pairpix=np.zeros((naxis,2,pix.size))
    pairpix[0,0]=img.pix[1:-1,2:][selection].ravel()
    pairpix[0,1]=img.pix[1:-1,:-2][selection].ravel()
    pairpix[1,0]=img.pix[2:,1:-1][selection].ravel()
    pairpix[1,1]=img.pix[:-2,1:-1][selection].ravel()
    pairpix[2,0]=img.pix[2:,2:][selection].ravel()
    pairpix[2,1]=img.pix[:-2,:-2][selection].ravel()
    pairpix[3,0]=img.pix[2:,:-2][selection].ravel()
    pairpix[3,1]=img.pix[:-2,2:][selection].ravel()
    pairivar=np.zeros((naxis,2,pix.size))
    pairivar[0,0]=(img.ivar[1:-1,2:][selection]*(img.mask[1:-1,2:][selection]==0)).ravel()
    pairivar[0,1]=(img.ivar[1:-1,:-2][selection]*(img.mask[1:-1,:-2][selection]==0)).ravel()
    pairivar[1,0]=(img.ivar[2:,1:-1][selection]*(img.mask[2:,1:-1][selection]==0)).ravel()
    pairivar[1,1]=(img.ivar[:-2,1:-1][selection]*(img.mask[:-2,1:-1][selection]==0)).ravel()
    pairivar[2,0]=(img.ivar[2:,2:][selection]*(img.mask[2:,2:][selection]==0)).ravel()
    pairivar[2,1]=(img.ivar[:-2,:-2][selection]*(img.mask[:-2,:-2][selection]==0)).ravel()
    pairivar[3,0]=(img.ivar[2:,:-2][selection]*(img.mask[2:,:-2][selection]==0)).ravel()
    pairivar[3,1]=(img.ivar[:-2,2:][selection]*(img.mask[:-2,2:][selection]==0)).ravel()

    # set to 0 pixel values with null ivar
    pairpix *= (pairivar>0)

    # back and sigmaback are the average values of each pair and their error
    # (same notation as SDSS code)
    tmp=np.sum(pairivar>0,axis=1).astype(float)
    tmp=(tmp>0)/(tmp+(tmp==0))
    back=np.sum(pairpix*(pairivar>0),axis=1)*tmp
    sigmaback=np.sqrt(np.sum((pairivar>0)/(pairivar+(pairivar==0)),axis=1))*tmp

    log.debug("mean pix = %f"%np.mean(pix))
    log.debug("mean back = %f"%np.mean(back))
    log.debug("mean sigmaback = %f"%np.mean(sigmaback))

    # first criterion, signal in pix must be significantly higher than neighbours (=back)
    # in all directions
    # JG comment : this does not look great for muon tracks that are perfectly aligned
    # with one the axis.
    # I change the algorithm to accept 3 out of 4 valid tests
    first_criterion=np.ones(pix.shape)
    nonullivar=pixivar>0
    tmp=np.zeros(pix.shape)
    tmp[nonullivar]=pix[nonullivar]-nsig/np.sqrt(pixivar[nonullivar])
    for a in range(naxis) :
        first_criterion += (tmp>back[a])
    first_criterion=(first_criterion>=3).astype(bool)

    # second criterion, rejected if at least for one axis
    # the values back are not consistent with PSF given the central
    # pixel value
    # here the number of sigmas is the parameter cfudge
    # c2fudge alters the PSF
    second_criterion=np.zeros(pix.shape).astype(bool)
    tmp=np.zeros(pix.shape)
    tmp[nonullivar]=pix[nonullivar]-cfudge/np.sqrt(pixivar[nonullivar])
    for a in range(naxis) :
        second_criterion |= ( tmp*c2fudge*psf[a] > ( back[a]+cfudge*sigmaback[a] ) )


    log.debug("npix selected                       = %d"%pix.size)
    log.debug("npix rejected 1st criterion         = %d"%np.sum(first_criterion))
    log.debug("npix rejected 1st and 2nd criterion = %d"%np.sum(first_criterion&second_criterion))

    # remap to original shape
    rejection=np.zeros(img.pix.shape).astype(bool)
    rejection[1:-1,1:-1][selection] = (first_criterion&second_criterion).reshape(img.pix[1:-1,1:-1][selection].shape)
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


    selection=((img.pix*np.sqrt(img.ivar)*(img.mask==0))[1:-1,1:-1]>nsig).astype(bool)
    rejected=reject_cosmic_rays_ala_sdss_single(img,selection,nsig=nsig,cfudge=cfudge,c2fudge=c2fudge)
    log.info("first pass: %d pixels rejected"%(np.sum(rejected)))

    tmpimg=copy.deepcopy(img)
    neighbors = np.zeros(rejected.shape).astype(bool)
    for iteration in range(niter) :

        if np.sum(rejected)==0 :
            break
        neighbors *= False
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
        tmpimg.ivar[rejected] = 0. # mask already rejected pixels for the calculation of the background of the neighbors

        # rerun with much more strict cuts
        newrejected=reject_cosmic_rays_ala_sdss_single(tmpimg,neighbors[1:-1,1:-1],nsig=3.,cfudge=0.,c2fudge=1.)
        log.info("at iter %d: %d new pixels rejected"%(iteration,np.sum(newrejected)))
        if np.sum(newrejected)<2 :
            break
        rejected = rejected|newrejected



    if dilate :
        log.debug("dilating cosmic ray mask")
        # now apply the dilatation included in sdssproc.pro
        # in IDL it is crmask = dilate(crmask, replicate(1,3,3))
        tmp=rejected.copy()
        rejected[1:-1,1:-1] |= tmp[1:-1,2:]
        rejected[1:-1,1:-1] |= tmp[1:-1,:-2]
        rejected[1:-1,1:-1] |= tmp[2:,1:-1]
        rejected[1:-1,1:-1] |= tmp[:-2,1:-1]
        rejected[1:-1,1:-1] |= tmp[2:,2:]
        rejected[1:-1,1:-1] |= tmp[:-2,:-2]
        rejected[1:-1,1:-1] |= tmp[2:,:-2]
        rejected[1:-1,1:-1] |= tmp[:-2,2:]

    log.info("end : %s pixels rejected"%(np.sum(rejected)))
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
