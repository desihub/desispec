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
from desispec.joincosmics import RepairMask


# Module-level object for mask repair, declared here to reduce instantiation
# overhead. Use 11x11-pixel selection elements to patch holes in the mask.
# However in restricted environments, such as ReadTheDocs, this throws
# an exception.
try:
    repair_mask = RepairMask(11,11)
except TypeError:
    repair_mask = None


def reject_cosmic_rays_1d(frame,nsig=3,psferr=0.05) :
    """Use resolution matrix in frame to detect spikes in the spectra that
    are narrower than the PSF, and mask them"""
    log=get_logger()

    log.info("subtract continuum to flux")
    tflux=np.zeros(frame.flux.shape)
    for fiber in range(frame.nspec) :
        tflux[fiber]=frame.flux[fiber]-scipy.ndimage.median_filter(frame.flux[fiber],200,mode='constant')
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

@numba.jit(nopython=True)
def dilate_numba(input_boolean_array,include_input=False) :
    output_boolean_array = np.zeros(input_boolean_array.shape, input_boolean_array.dtype)
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



@numba.jit(nopython=True)
def _reject_cosmic_rays_ala_sdss_single_numba(pix,ivar,selection,psf_gradients,nsig,cfudge,c2fudge) :
    """Cosmic ray rejection following the implementation in SDSS/BOSS.
    (see idlutils/src/image/reject_cr_psf.c and idlutils/pro/image/reject_cr.pro)

    This routine is a single call, similar to IDL routine reject_cr_single
    Called by reject_cosmic_rays_ala_sdss with an iteration

    Input is a pre-processed image : desispec.Image
    Ouput is a rejection mask of the same size as the image

    This routine is much faster than _reject_cosmic_rays_ala_sdss_single
    (if you have numba installed, otherwise it is catastrophically slower)

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
    dd = np.zeros((naxis,2),dtype=type(n0))
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

    rejection=np.zeros(pix.shape,dtype=type(True))

    for i0 in range(1,n0-1) :
        for i1 in range(1,n1-1) :

            central_pix_ivar=ivar[i0,i1]
            if (not selection[i0,i1]) or central_pix_ivar<=0 : continue

            # first criterion, signal in pix must be significantly higher than neighbors
            # in all directions
            # JG comment : this does not look great for muon tracks that are perfectly aligned
            # with one the axis. I change the algorithm to accept 2 out of 4 valid tests
            first_criterion=0

            # second criterion, rejected if at least for one axis
            # the neighbors average value are not consistent with PSF given the central pixel value
            # here the number of sigmas is the parameter cfudge
            # c2fudge alters the PSF
            second_criterion=False

            central_pix_val=pix[i0,i1]
            central_pix_err=1/np.sqrt(central_pix_ivar)

            # loop on axis
            for a in range(naxis) :

                # the offsets
                d0=dd[a,0]
                d1=dd[a,1]

                neighboring_pix_val=0.
                neighboring_pix_err=0.

                # compute average value on both sides of central pix
                for signe in [-1,1] :
                    tmp_ivar = ivar[i0+signe*d0,i1+signe*d1]
                    if tmp_ivar > 0 :
                        neighboring_pix_val  += pix[i0+signe*d0,i1+signe*d1]
                        neighboring_pix_err  += 1/tmp_ivar
                    else : # replace it by the central pixel value
                        neighboring_pix_val  += central_pix_val
                        neighboring_pix_err  += central_pix_err**2

                neighboring_pix_val  *= 0.5 # average value
                neighboring_pix_err   = np.sqrt(neighboring_pix_err)*0.5 # uncertainty on average value

                first_criterion += (central_pix_val>(neighboring_pix_val+nsig*central_pix_err))
                second_criterion |= (((central_pix_val-cfudge*central_pix_err)*c2fudge*psf_gradients[a]) > ( neighboring_pix_val+cfudge*neighboring_pix_err ))

            rejection[i0,i1] = ( (first_criterion>=2) & second_criterion )

    return rejection

def _reject_cosmic_rays_ala_sdss_single(pix,ivar,selection,psf_gradients,nsig,cfudge,c2fudge) :
    """Cosmic ray rejection following the implementation in SDSS/BOSS.
    (see idlutils/src/image/reject_cr_psf.c and idlutils/pro/image/reject_cr.pro)

    This routine is a single call, similar to IDL routine reject_cr_single
    Called by reject_cosmic_rays_ala_sdss with an iteration

    Input is a pre-processed image : desispec.Image
    Ouput is a rejection mask of the same size as the image

    A faster version of this routine using numba in implemented in
    _reject_cosmic_rays_ala_sdss_single_numba

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
    for a in range(naxis) :
        for i in range(2) :
            jj=(pairivar[a,i]==0)
            pairpix[a,i,jj]=tpix[jj] # replace null ivar pair pixel value with central value
            pairivar[a,i,jj]=tpixivar[jj]

    back=np.sum(pairpix*(pairivar>0),axis=1)*0.5
    sigmaback=np.sqrt(np.sum((pairivar>0)/(pairivar+(pairivar==0)),axis=1))*0.5

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
    first_criterion=(first_criterion>=3)

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

def reject_cosmic_rays_ala_sdss(img,nsig=6.,cfudge=3.,c2fudge=0.5,niter=6,dilate=True) :
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

    # those gradients have been computed using the code desi_compute_psf_gradients
    # 2019-04-17: switch from sharpest psf among all fibers and wavelength to mean psf on the CCD
    # this is to avoid being sensitive to noisy psf at short or long wavelength.
    # An analysis of real darks combined with real continuum suggest a best value for c2fudge=0.8
    # See https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=4893
    # A new analysis using arc lines gives a conservative value c2fudge=0.5
    # see https://github.com/desihub/desispec/issues/762

    band=img.camera[0].lower()
    if band == 'b':
        # desi_compute_psf_gradients -i $DESI_SPECTRO_CALIB/spec/sp2/psf-b2-20190308.fit
        psf_gradients=np.array([0.758221,0.771945,0.570183,0.592199])
    elif band == 'r':
        # desi_compute_psf_gradients -i $DESI_SPECTRO_CALIB/spec/sp2/psf-r2-20190308.fits
        psf_gradients=np.array([0.819245,0.847529,0.617514,0.656629])
    elif band == 'z':
        # desi_compute_psf_gradients -i $DESI_SPECTRO_CALIB/spec/sp2/psf-z2-20190308.fits
        psf_gradients=np.array([0.829552,0.862828,0.633424,0.664144])
    else :
        log.error("do not have psf info for band='%s'"%band)
        raise KeyError("do not have psf info for band='%s'"%band)

    selection = ((img.pix*np.sqrt(tivar))>nsig)

    use_numba = True

    if use_numba :
        rejected   = _reject_cosmic_rays_ala_sdss_single_numba(img.pix,tivar,selection,psf_gradients,nsig=nsig,cfudge=cfudge,c2fudge=c2fudge)
    else :
        rejected  = _reject_cosmic_rays_ala_sdss_single(img.pix,tivar,selection,psf_gradients,nsig=nsig,cfudge=cfudge,c2fudge=c2fudge)


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
                newrejected=_reject_cosmic_rays_ala_sdss_single_numba(img.pix,tivar,neighbors,psf_gradients,nsig=3.,cfudge=0.,c2fudge=c2fudge)
            else :
                newrejected=_reject_cosmic_rays_ala_sdss_single(img.pix,tivar,neighbors,psf_gradients,nsig=3.,cfudge=0.,c2fudge=c2fudge)

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

        # Apply binary closure repair defined in joincosmics.
        log.debug('Repairing gaps in cosmic ray mask')
        rejected = repair_mask.repair(rejected)

    t1=time.time()
    log.info("end : {} pixels rejected in {:3.1f} sec".format(np.sum(rejected),t1-t0))
    return rejected

def reject_cosmic_rays(img,nsig=5.,cfudge=3.,c2fudge=0.9,niter=100,dilate=True) :
    """Cosmic ray rejection
    Input is a pre-processed image : desispec.Image
    The image mask is modified

    Args:
       img: input desispec.Image

    """
    rejected=reject_cosmic_rays_ala_sdss(img,nsig=nsig,cfudge=cfudge,c2fudge=c2fudge,niter=niter,dilate=dilate)
    img.mask[rejected] |= ccdmask.COSMIC
