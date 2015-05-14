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

import scipy,scipy.sparse
import sys

def compute_sky(spectra, nsig_clipping=4.) :
    """Compute a sky model.

    Input has to correspond to sky fibers only.
    Input flux are expected to be flatfielded!
    We don't check this in this routine.

    args:
        spectra : Spectra object of sky fibers, which includes attributes
          - wave : 1D wavelength grid in Angstroms
          - flux : 2D flux[nspec, nwave] density (only sky fibers)
          - ivar : 2D inverse variance of flux (only sky fibers)
          - resolution_data : 3D[nspec, ndiag, nwave]  (only sky fibers)
        nsig_clipping : [optional] sigma clipping value for outlier rejection

    returns SkyModel object with attributes
      - skyflux : 1D[nwave] deconvolved skyflux
      - ivar : inverse variance of that skyflux
      - mask : 0=ok >0 if problems
      - cskyflux :  1D[nwave] convolved skyflux at average resolution
      - cskyivar :  1D[nwave] convolved skyflux inverse variance
    """

    log=get_logger()
    log.info("starting")

    nwave=spectra.nwave
    nfibers=spectra.nspec
    current_ivar=spectra.ivar.copy()

    sqrtw=np.sqrt(current_ivar)
    sqrtwflux=sqrtw*spectra.flux

    chi2=np.zeros(spectra.flux.shape)

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
            R = spectra.R[fiber]

            # diagonal sparse matrix with content = sqrt(ivar)
            SD.setdiag(sqrtw[fiber])

            sqrtwR = SD*R # each row r of R is multiplied by sqrtw[r]

            A = A+(sqrtwR.T*sqrtwR).tocsr()
            B += sqrtwR.T*sqrtwflux[fiber]

        log.info("iter %d solving"%iteration)

        skyflux=cholesky_solve(A.todense(),B)

        log.info("iter %d compute chi2"%iteration)

        for fiber in range(nfibers) :

            S = spectra.R[fiber].dot(skyflux)
            chi2[fiber]=current_ivar[fiber]*(spectra.flux[fiber]-S)**2

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

    skyvar=np.diagonal(skycovar)
    skyivar=(skyvar>0)/(skyvar+(skyvar==0))

    # we also want to save the convolved sky and sky variance
    # this might be handy

    # first compute average resolution
    mean_res_data=np.mean(spectra.resolution_data,axis=0)
    R = Resolution(mean_res_data)
    # compute convolved sky and ivar
    cskyflux=R.dot(skyflux)
    cskycovar=R.dot(skycovar).dot(R.T.todense())
    cskyvar=np.diagonal(cskycovar)
    cskyivar=(cskyvar>0)/(cskyvar+(cskyvar==0))



    # need to do better here
    mask=(skyvar>0).astype(long)  # SOMEONE CHECK THIS !

    return SkyModel(spectra.wave.copy(), skyflux, skyivar, mask, cskyflux, cskyivar)

class SkyModel(object):
    def __init__(self, wave, flux, ivar, mask, cflux, civar):
        """Create SkyModel object
        
        Args:
            wave  : 1D[nwave] wavelength in Angstroms
            flux  : 1D[nwave] deconvolved skyflux
            ivar  : inverse variance of that skyflux
            mask  : 0=ok >0 if problems
            cflux : 1D[nwave] convolved skyflux at average resolution
            civar : 1D[nwave] convolved skyflux inverse variance
            
        All input arguments become attributes
        """
        if wave.ndim != 1:  raise ValueError('wave should be 1D')
        if flux.ndim != 1:  raise ValueError('flux should be 1D')
        if ivar.ndim != 1:  raise ValueError('ivar should be 1D')
        if mask.ndim != 1:  raise ValueError('mask should be 1D')
        if cflux.ndim != 1: raise ValueError('cflux should be 1D')
        if civar.ndim != 1: raise ValueError('civar should be 1D')
        self.nwave = len(wave)
        if len(flux) != self.nwave: raise ValueError('len(flux) != nwave')
        if len(ivar) != self.nwave: raise ValueError('len(ivar) != nwave')
        if len(mask) != self.nwave: raise ValueError('len(mask) != nwave')
        if len(cflux) != self.nwave: raise ValueError('len(cflux) != nwave')
        if len(civar) != self.nwave: raise ValueError('len(civar) != nwave')
        
        self.wave = wave
        self.flux = flux
        self.ivar = ivar
        self.mask = mask
        self.cflux = cflux
        self.civar = civar


def subtract_sky(spectra, skymodel) :
    """Subtract skymodel from spectra, accounting for per-fiber resolution
    
    Alters spectra.flux and spectra.ivar
    """
    log=get_logger()
    log.info("starting")

    # check same wavelength, die if not the case
    if not np.allclose(spectra.wave, skymodel.wave):
        message = "spectra and sky not on same wavelength grid"
        log.error(message)
        raise ValueError(message)

    nwave=spectra.nwave
    nfibers=spectra.nspec

    for fiber in range(nfibers) :

        #if fiber%10==0 :
        #    log.info("fiber %d"%fiber)

        spectra.flux[fiber] -= spectra.R[fiber].dot(skymodel.flux)

        # deal with variance
        selection=np.where((spectra.ivar[fiber]>0)&(skymodel.civar>0))[0]
        if selection.size==0 :
            continue

        spectra.ivar[fiber,selection]=1./(1./spectra.ivar[fiber,selection]+1./skymodel.civar[selection])

    log.info("done")
