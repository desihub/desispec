#!/usr/bin/env python

"""
desispec.cosmics
============

Utility functions to find cosmic rays
"""


from desispec.log import get_logger
import numpy as np
import math

def reject_cosmic_rays_ala_sdss(img,psf_sigma_pix=1.,nsig=6.,cfudge=3.,c2fudge=1.) :
    """Cosmic ray rejection following the implementation in SDSS/BOSS.
    (see idlutils/src/image/reject_cr_psf.c)
    
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
    log.info("starting with nsig=%2.1f cfudge=%2.1f c2fudge=%2.1f"%(nsig,cfudge,c2fudge))
    
    # assume a Gaussian PSF with sigma_pix
    # psf is the ratio of pixel intensity between center and offsets of 1 or sqrt(2) pixels depending
    # on the direction.
    # JG : we could use a real PSF but this is very likely not necessary
    # JG : we can imagine using a sigma_pix as an image to account for PSF variation in CCD
    # JG : we can directly apply the "cfudge2" coefficient in the PSF here
    naxis=4
    psf=np.zeros((naxis))
    psf[0]=math.exp(-1./(2*psf_sigma_pix**2))
    psf[1]=math.exp(-1./(2*psf_sigma_pix**2))
    psf[2]=math.exp(-2./(2*psf_sigma_pix**2))
    psf[3]=math.exp(-2./(2*psf_sigma_pix**2))
    


    # goodback is True on valid pixels in each pair
    # there are four axis for the pairs and 2 pixels per pair
    # (same notation as reject_cr_psf.c)
    goodback=np.zeros((naxis,2,img.pix.shape[0],img.pix.shape[1])) 
    goodback[0,0,1:-1,1:-1]=img.ivar[1:-1,2:]>0
    goodback[0,1,1:-1,1:-1]=img.ivar[1:-1,:-2]>0
    goodback[1,0,1:-1,1:-1]=img.ivar[2:,1:-1]>0
    goodback[1,1,1:-1,1:-1]=img.ivar[:-2,1:-1]>0
    goodback[2,0,1:-1,1:-1]=img.ivar[2:,2:]>0
    goodback[2,1,1:-1,1:-1]=img.ivar[:-2,:-2]>0
    goodback[3,0,1:-1,1:-1]=img.ivar[2:,:-2]>0
    goodback[3,1,1:-1,1:-1]=img.ivar[:-2,2:]>0
    
    # keep only valid pixels
    #goodpix=(img.ivar>0)*img.pix*(img.mask==0)
    #var=(img.ivar>0)*(img.mask==0)/(img.ivar*(img.ivar>0)+(img.ivar<=0))
    log.warning("temporary hack: ignore the mask to find cosmics")
    goodpix=(img.ivar>0)*img.pix
    var=(img.ivar>0)/(img.ivar*(img.ivar>0)+(img.ivar<=0))
    
    

    # back is the average pix value in the pair of pixels on each side of the pixel of interest
    # there are four directions for the pairs
    # sigmaback is its uncertainty 
    # (same notation as reject_cr_psf.c)
    back=np.zeros((naxis,img.pix.shape[0],img.pix.shape[1]))
    sigmaback=np.zeros((naxis,img.pix.shape[0],img.pix.shape[1]))
    
    tmp=np.sum(goodback[0],axis=0)[1:-1,1:-1]
    back[0,1:-1,1:-1]=(tmp>0)*(goodpix[1:-1,2:]+goodpix[1:-1,:-2])/(tmp*(tmp>0)+(tmp<=0))
    sigmaback[0,1:-1,1:-1]=(tmp>0)*np.sqrt(var[1:-1,2:]+var[1:-1,:-2])/(tmp*(tmp>0)+(tmp<=0))
    
    tmp=np.sum(goodback[1],axis=0)[1:-1,1:-1]
    back[1,1:-1,1:-1]=(tmp>0)*(goodpix[2:,1:-1]+goodpix[:-2,1:-1])/(tmp*(tmp>0)+(tmp<=0))
    sigmaback[1,1:-1,1:-1]=(tmp>0)*np.sqrt(var[2:,1:-1]+var[:-2,1:-1])/(tmp*(tmp>0)+(tmp<=0))

    tmp=np.sum(goodback[2],axis=0)[1:-1,1:-1]
    back[2,1:-1,1:-1]=(tmp>0)*(goodpix[2:,2:]+goodpix[:-2,:-2])/(tmp*(tmp>0)+(tmp<=0))
    sigmaback[2,1:-1,1:-1]=(tmp>0)*np.sqrt(var[2:,2:]+var[:-2,:-2])/(tmp*(tmp>0)+(tmp<=0))
    
    tmp=np.sum(goodback[3],axis=0)[1:-1,1:-1]
    back[3,1:-1,1:-1]=(tmp>0)*(goodpix[2:,:-2]+goodpix[:-2,2:])/(tmp*(tmp>0)+(tmp<=0))
    sigmaback[3,1:-1,1:-1]=(tmp>0)*np.sqrt(var[2:,:-2]+var[:-2,2:])/(tmp*(tmp>0)+(tmp<=0))
    
    
    
    # first criterion, signal in pix must be significantly higher than neighbours (=back)
    # in all directions 
    # JG comment : this is not great for muon tracks for instance where one axis 
    # will likely not pass the test
    first_criterion=np.ones(img.pix.shape)
    for a in range(naxis) :
        first_criterion *= ( goodpix>(back[a]+nsig*np.sqrt(var)) )
    
    # second criterion, rejected if at least for one axis
    # the values back are not consistent with PSF given the central
    # pixel value
    # here the number of sigmas is the parameter cfudge
    # c2fudge alters the PSF
    second_criterion=np.zeros(img.pix.shape)
    for a in range(naxis) :
        second_criterion += ( ( goodpix-cfudge*np.sqrt(var) )*c2fudge*psf[a] > ( back[a]+cfudge*sigmaback[a] ) )
    
    log.info("done")
    return first_criterion*(second_criterion>0)

