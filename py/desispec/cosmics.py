#!/usr/bin/env python

"""
desispec.cosmics
============

Utility functions to find cosmic rays
"""


from desispec.log import get_logger
import numpy as np
import math
import copy
#import sys # for debug 
#import pyfits # for debug

def reject_cosmic_rays_ala_sdss_single(img,psf_sigma_pix=1.,nsig=6.,cfudge=3.,c2fudge=1.) :
    """Cosmic ray rejection following the implementation in SDSS/BOSS.
    (see idlutils/src/image/reject_cr_psf.c and idlutils/pro/image/reject_cr.pro)
    
    This routine is a single call, similar to IDL routine reject_cr_single
    Called by reject_cosmic_rays_ala_sdss with an iteration
    
    Input is a pre-processed image : desispec.Image
    Ouput is a rejection mask of the same size as the image
    
    Args:
       img: input desispec.Image
       psf_sigma_pix: sigma of Gaussian PSF in pixel units (can do better if needed)
       nsig: number of sigma above background required
       cfudge: number of sigma inconsistent with PSF required
       c2fudge:  fudge factor applied to PSF
    """
    log=get_logger()
    log.debug("starting with nsig=%2.1f cfudge=%2.1f c2fudge=%2.1f"%(nsig,cfudge,c2fudge))
    
    # assume a Gaussian PSF with sigma_pix
    # psf is the ratio of pixel intensity between center and offsets of 1 or sqrt(2) pixels depending
    # on the direction.
    # JG : we could use a real PSF but this is very likely not necessary, for instance
    # we could use for sigma_pix an image to account for PSF variation in CCD
    
    naxis=4
    psf=np.zeros((naxis))
    psf[0]=math.exp(-1./(2*psf_sigma_pix**2))
    psf[1]=math.exp(-1./(2*psf_sigma_pix**2))
    psf[2]=math.exp(-2./(2*psf_sigma_pix**2))
    psf[3]=math.exp(-2./(2*psf_sigma_pix**2))
    
    # we preselect pixels above threshold to try to go as fast as possible with python
    selection=((img.pix*np.sqrt(img.ivar)*(img.mask==0))[1:-1,1:-1]>nsig).astype(bool)
    
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
    # JG comment : this does not look great for muon tracks for instance where one axis 
    # will likely not pass the test, but it works ... we can study this latter 
    first_criterion=np.ones(pix.shape).astype(bool)
    tmp=pix-nsig/np.sqrt(pixivar)
    for a in range(naxis) :
        first_criterion &= (tmp>back[a])
    
    # second criterion, rejected if at least for one axis
    # the values back are not consistent with PSF given the central
    # pixel value
    # here the number of sigmas is the parameter cfudge
    # c2fudge alters the PSF
    second_criterion=np.zeros(pix.shape).astype(bool)
    tmp=pix-cfudge/np.sqrt(pixivar)
    for a in range(naxis) :
        second_criterion |= ( tmp*c2fudge*psf[a] > ( back[a]+cfudge*sigmaback[a] ) )
        
        
    log.debug("npix selected                       = %d"%pix.size)
    log.debug("npix rejected 1st criterion         = %d"%np.sum(first_criterion))
    log.debug("npix rejected 1st and 2nd criterion = %d"%np.sum(first_criterion&second_criterion))
    
    # remap to original shape 
    rejection=np.zeros(img.pix.shape).astype(bool)
    rejection[1:-1,1:-1][selection] = (first_criterion&second_criterion).reshape(img.pix[1:-1,1:-1][selection].shape)
    return rejection

def reject_cosmic_rays_ala_sdss(img,psf_sigma_pix=1.,nsig=6.,cfudge=3.,c2fudge=0.8,niter=6,dilate=True) :
    """Cosmic ray rejection following the implementation in SDSS/BOSS.
    (see idlutils/src/image/reject_cr_psf.c and idlutils/pro/image/reject_cr.pro)
    
    This routine is calling several times reject_cosmic_rays_ala_sdss_single, 
    similar to IDL routine reject_cr.
    There is an optionnal dilatation of the mask by one pixel, as done in sdssproc.pro for SDSS
    
    Input is a pre-processed image : desispec.Image
    Ouput is a rejection mask of the same size as the image
    
    Args:
       img: input desispec.Image
       psf_sigma_pix: sigma of Gaussian PSF in pixel units (can do better if needed)
       nsig: number of sigma above background required
       cfudge: number of sigma inconsistent with PSF required
       c2fudge:  fudge factor applied to PSF
       niter: number of iterations on neighboring pixels of rejected pixels
       dilate: force +1 pixel dilation of rejection mask 
    """
    log=get_logger()
    log.info("starting with nsig=%2.1f cfudge=%2.1f c2fudge=%2.1f"%(nsig,cfudge,c2fudge))

    
   
    rejected=reject_cosmic_rays_ala_sdss_single(img,psf_sigma_pix=psf_sigma_pix,nsig=nsig,cfudge=cfudge,c2fudge=c2fudge)
    log.info("first pass: %d pixels rejected"%(np.sum(rejected)))
    
    tmpimg=copy.deepcopy(img)
    neighbors = np.zeros(rejected.shape).astype(bool)
    for iteration in range(niter) :
          
        if np.sum(rejected)==0 :
            break
        neighbors   *= 0
        tmpimg.ivar *= 0
        neighbors[1:,:]  |= rejected[:-1,:]
        neighbors[:-1,:] |= rejected[1:,:]
        neighbors[:,1:]  |= rejected[:,:-1]
        neighbors[:,:-1] |= rejected[:,1:]
        # adding diagonals (not in SDSS version)
        neighbors[1:,1:]  |= rejected[:-1,:-1]
        neighbors[:-1,:-1]  |= rejected[1:,1:]
        neighbors[1:,:-1]  |= rejected[:-1,1:]
        neighbors[:-1,1:]  |= rejected[1:,:-1]
        
        tmpimg.ivar[neighbors]=img.ivar[neighbors]*(rejected[neighbors]==0) 
        
        newrejected=reject_cosmic_rays_ala_sdss_single(tmpimg,psf_sigma_pix,nsig=nsig,cfudge=0.,c2fudge=1.)
        log.info("at iter %d: %d new pixels rejected"%(iteration,np.sum(newrejected)))
        if np.sum(newrejected)<2 :
            break
        rejected = rejected|newrejected
    
    
        
    if dilate :    
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
