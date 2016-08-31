"""
desispec.sky
============

Utility functions to compute a sky model and subtract it.
"""


import numpy as np
from desispec.resolution import Resolution
from desispec.linalg import cholesky_solve
from desispec.linalg import cholesky_solve_and_invert
from desispec.linalg import spline_fit
from desispec.log import get_logger
from desispec import util

from desiutil import stats as dustat

import scipy,scipy.sparse,scipy.stats,scipy.ndimage
import sys

def compute_sky(frame, nsig_clipping=4.) :
    """Compute a sky model.

    Input has to correspond to sky fibers only.
    Input flux are expected to be flatfielded!
    We don't check this in this routine.

    Args:
        frame : Frame object, which includes attributes
          - wave : 1D wavelength grid in Angstroms
          - flux : 2D flux[nspec, nwave] density
          - ivar : 2D inverse variance of flux
          - mask : 2D inverse mask flux (0=good)
          - resolution_data : 3D[nspec, ndiag, nwave]  (only sky fibers)
        nsig_clipping : [optional] sigma clipping value for outlier rejection

    returns SkyModel object with attributes wave, flux, ivar, mask
    """

    log=get_logger()
    log.info("starting")

    # Grab sky fibers on this frame
    skyfibers = np.where(frame.fibermap['OBJTYPE'] == 'SKY')[0]
    assert np.max(skyfibers) < 500  #- indices, not fiber numbers

    nwave=frame.nwave
    nfibers=len(skyfibers)

    current_ivar=frame.ivar[skyfibers].copy()
    flux = frame.flux[skyfibers]
    Rsky = frame.R[skyfibers]

    sqrtw=np.sqrt(current_ivar)
    sqrtwflux=sqrtw*flux

    chi2=np.zeros(flux.shape)

    #debug
    #nfibers=min(nfibers,2)

    nout_tot=0
    for iteration in range(20) :

        A=scipy.sparse.lil_matrix((nwave,nwave)).tocsr()
        B=np.zeros((nwave))
        # diagonal sparse matrix with content = sqrt(ivar)*flat of a given fiber
        SD=scipy.sparse.lil_matrix((nwave,nwave))
        # loop on fiber to handle resolution
        for fiber in range(nfibers) :
            if fiber%10==0 :
                log.info("iter %d fiber %d"%(iteration,fiber))
            R = Rsky[fiber]

            # diagonal sparse matrix with content = sqrt(ivar)
            SD.setdiag(sqrtw[fiber])

            sqrtwR = SD*R # each row r of R is multiplied by sqrtw[r]

            A = A+(sqrtwR.T*sqrtwR).tocsr()
            B += sqrtwR.T*sqrtwflux[fiber]

        log.info("iter %d solving"%iteration)

        skyflux=cholesky_solve(A.todense(),B)

        log.info("iter %d compute chi2"%iteration)

        for fiber in range(nfibers) :

            S = Rsky[fiber].dot(skyflux)
            chi2[fiber]=current_ivar[fiber]*(flux[fiber]-S)**2

        log.info("rejecting")

        nout_iter=0
        if iteration<1 :
            # only remove worst outlier per wave
            # apply rejection iteratively, only one entry per wave among fibers
            # find waves with outlier (fastest way)
            nout_per_wave=np.sum(chi2>nsig_clipping**2,axis=0)
            selection=np.where(nout_per_wave>0)[0]
            for i in selection :
                worst_entry=np.argmax(chi2[:,i])
                current_ivar[worst_entry,i]=0
                sqrtw[worst_entry,i]=0
                sqrtwflux[worst_entry,i]=0
                nout_iter += 1

        else :
            # remove all of them at once
            bad=(chi2>nsig_clipping**2)
            current_ivar *= (bad==0)
            sqrtw *= (bad==0)
            sqrtwflux *= (bad==0)
            nout_iter += np.sum(bad)

        nout_tot += nout_iter

        sum_chi2=float(np.sum(chi2))
        ndf=int(np.sum(chi2>0)-nwave)
        chi2pdf=0.
        if ndf>0 :
            chi2pdf=sum_chi2/ndf
        log.info("iter #%d chi2=%f ndf=%d chi2pdf=%f nout=%d"%(iteration,sum_chi2,ndf,chi2pdf,nout_iter))

        if nout_iter == 0 :
            break

    log.info("nout tot=%d"%nout_tot)


    # solve once again to get deconvolved sky variance
    skyflux,skycovar=cholesky_solve_and_invert(A.todense(),B)

    #- sky inverse variance, but incomplete and not needed anyway
    # skyvar=np.diagonal(skycovar)
    # skyivar=(skyvar>0)/(skyvar+(skyvar==0))

    # Use diagonal of skycovar convolved with mean resolution of all fibers
    # first compute average resolution
    mean_res_data=np.mean(frame.resolution_data,axis=0)
    R = Resolution(mean_res_data)
    # compute convolved sky and ivar
    cskycovar=R.dot(skycovar).dot(R.T.todense())
    cskyvar=np.diagonal(cskycovar)
    cskyivar=(cskyvar>0)/(cskyvar+(cskyvar==0))

    # convert cskyivar to 2D; today it is the same for all spectra,
    # but that may not be the case in the future
    cskyivar = np.tile(cskyivar, frame.nspec).reshape(frame.nspec, nwave)

    # Convolved sky
    cskyflux = np.zeros(frame.flux.shape)
    for i in range(frame.nspec):
        cskyflux[i] = frame.R[i].dot(skyflux)

    # need to do better here
    mask = (cskyivar==0).astype(np.uint32)

    return SkyModel(frame.wave.copy(), cskyflux, cskyivar, mask,
                    nrej=nout_tot)

class SkyModel(object):
    def __init__(self, wave, flux, ivar, mask, header=None, nrej=0):
        """Create SkyModel object

        Args:
            wave  : 1D[nwave] wavelength in Angstroms
            flux  : 2D[nspec, nwave] sky model to subtract
            ivar  : 2D[nspec, nwave] inverse variance of the sky model
            mask  : 2D[nspec, nwave] 0=ok or >0 if problems; 32-bit
            header : (optional) header from FITS file HDU0
            nrej : (optional) Number of rejected pixels in fit

        All input arguments become attributes
        """
        assert wave.ndim == 1
        assert flux.ndim == 2
        assert ivar.shape == flux.shape
        assert mask.shape == flux.shape

        self.nspec, self.nwave = flux.shape
        self.wave = wave
        self.flux = flux
        self.ivar = ivar
        self.mask = util.mask32(mask)
        self.header = header
        self.nrej = nrej


def subtract_sky(frame, skymodel) :
    """Subtract skymodel from frame, altering frame.flux, .ivar, and .mask

    Args:
        frame : desispec.Frame object
        skymodel : desispec.SkyModel object
    """
    assert frame.nspec == skymodel.nspec
    assert frame.nwave == skymodel.nwave

    log=get_logger()
    log.info("starting")

    # check same wavelength, die if not the case
    if not np.allclose(frame.wave, skymodel.wave):
        message = "frame and sky not on same wavelength grid"
        log.error(message)
        raise ValueError(message)

    frame.flux -= skymodel.flux
    frame.ivar = util.combine_ivar(frame.ivar, skymodel.ivar)
    frame.mask |= skymodel.mask

    log.info("done")


def qa_skysub(param, frame, skymodel, quick_look=False):
    """Calculate QA on SkySubtraction

    Note: Pixels rejected in generating the SkyModel (as above), are
    not rejected in the stats calculated here.  Would need to carry
    along current_ivar to do so.

    Args:
        param : dict of QA parameters
        frame : desispec.Frame object
        skymodel : desispec.SkyModel object
        quick_look : bool, optional
          If True, do QuickLook specific QA (or avoid some)
    Returns:
        qadict: dict of QA outputs
          Need to record simple Python objects for yaml (str, float, int)
    """
    log=get_logger()

    # Output dict
    qadict = {}
    qadict['NREJ'] = int(skymodel.nrej)

    # Grab sky fibers on this frame
    skyfibers = np.where(frame.fibermap['OBJTYPE'] == 'SKY')[0]
    assert np.max(skyfibers) < 500  #- indices, not fiber numbers
    nfibers=len(skyfibers)
    qadict['NSKY_FIB'] = int(nfibers)

    current_ivar=frame.ivar[skyfibers].copy()
    flux = frame.flux[skyfibers]

    # Subtract
    res = flux - skymodel.flux[skyfibers] # Residuals
    res_ivar = util.combine_ivar(current_ivar, skymodel.ivar[skyfibers])

    # Chi^2 and Probability
    chi2_fiber = np.sum(res_ivar*(res**2),1)
    chi2_prob = np.zeros(nfibers)
    for ii in range(nfibers):
        # Stats
        dof = np.sum(res_ivar[ii,:] > 0.)
        chi2_prob[ii] = scipy.stats.chisqprob(chi2_fiber[ii], dof)
    # Bad models
    qadict['NBAD_PCHI'] = int(np.sum(chi2_prob < param['PCHI_RESID']))
    if qadict['NBAD_PCHI'] > 0:
        log.warning("Bad Sky Subtraction in {:d} fibers".format(
                qadict['NBAD_PCHI']))

    # Median residual
    qadict['MED_RESID'] = float(np.median(res)) # Median residual (counts)
    log.info("Median residual for sky fibers = {:g}".format(
        qadict['MED_RESID']))

    # Residual percentiles
    perc = dustat.perc(res, per=param['PER_RESID'])
    qadict['RESID_PER'] = [float(iperc) for iperc in perc]

    # Mean Sky Continuum from all skyfibers
    # need to limit in wavelength?

    if quick_look:
        continuum=scipy.ndimage.filters.median_filter(flux,200) # taking 200 bins (somewhat arbitrarily)
        mean_continuum=np.zeros(flux.shape[1])
        for ii in range(flux.shape[1]):
            mean_continuum[ii]=np.mean(continuum[:,ii])
        qadict['MEAN_CONTIN'] = mean_continuum

    # Median Signal to Noise on sky subtracted spectra
    # first do the subtraction:
    if quick_look:
        fframe=frame # make a copy
        sskymodel=skymodel # make a copy
        subtract_sky(fframe,sskymodel)
        medsnr=np.zeros(fframe.flux.shape[0])
        totsnr=np.zeros(fframe.flux.shape[0])
        for ii in range(fframe.flux.shape[0]):
            signalmask=fframe.flux[ii,:]>0
            # total snr considering bin by bin uncorrelated S/N
            snr=fframe.flux[ii,signalmask]*np.sqrt(fframe.ivar[ii,signalmask])
            medsnr[ii]=np.median(snr)
            totsnr[ii]=np.sqrt(np.sum(snr**2))
        qadict['MED_SNR']=medsnr  # for each fiber
        qadict['TOT_SNR']=totsnr  # for each fiber

    # Return
    return qadict
