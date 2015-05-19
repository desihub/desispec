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

import scipy,scipy.sparse
import sys

def compute_sky(spectra, fibermap, nsig_clipping=4.) :
    """Compute a sky model.

    Input has to correspond to sky fibers only.
    Input flux are expected to be flatfielded!
    We don't check this in this routine.

    args:
        spectra : Spectra object, which includes attributes
          - wave : 1D wavelength grid in Angstroms
          - flux : 2D flux[nspec, nwave] density
          - ivar : 2D inverse variance of flux
          - mask : 2D inverse mask flux (0=good)
          - resolution_data : 3D[nspec, ndiag, nwave]  (only sky fibers)
        fibermap : numpy table including OBJTYPE to know which fibers are SKY
        nsig_clipping : [optional] sigma clipping value for outlier rejection

    returns SkyModel object with attributes wave, flux, ivar, mask
    """

    log=get_logger()
    log.info("starting")

    skyfibers = np.where(fibermap["OBJTYPE"]=="SKY")[0]

    nwave=spectra.nwave
    nfibers=len(skyfibers)
    
    current_ivar=spectra.ivar[skyfibers].copy()
    flux = spectra.flux[skyfibers]
    Rsky = spectra.R[skyfibers]

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
    mean_res_data=np.mean(spectra.resolution_data,axis=0)
    R = Resolution(mean_res_data)
    # compute convolved sky and ivar
    cskycovar=R.dot(skycovar).dot(R.T.todense())
    cskyvar=np.diagonal(cskycovar)
    cskyivar=(cskyvar>0)/(cskyvar+(cskyvar==0))
    
    # convert cskyivar to 2D; today it is the same for all spectra,
    # but that may not be the case in the future
    cskyivar = np.tile(cskyivar, spectra.nspec).reshape(spectra.nspec, nwave)

    # Convolved sky
    cskyflux = np.zeros(spectra.flux.shape)
    for i in range(spectra.nspec):
        cskyflux[i] = spectra.R[i].dot(skyflux)

    # need to do better here
    mask = (cskyivar==0).astype(int)

    return SkyModel(spectra.wave.copy(), cskyflux, cskyivar, mask)

class SkyModel(object):
    def __init__(self, wave, flux, ivar, mask, header=None):
        """Create SkyModel object
        
        Args:
            wave  : 1D[nwave] wavelength in Angstroms
            flux  : 2D[nspec, nwave] sky model to subtract
            ivar  : 2D[nspec, nwave] inverse variance of the sky model
            mask  : 2D[nspec, nwave] 0=ok or >0 if problems
            header : (optional) header from FITS file HDU0
            
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
        self.mask = mask
        self.header = header


def subtract_sky(spectra, skymodel) :
    """Subtract skymodel from spectra, altering spectra.flux, .ivar, and .mask
    """
    assert spectra.nspec == skymodel.nspec
    assert spectra.nwave == skymodel.nwave

    log=get_logger()
    log.info("starting")

    # check same wavelength, die if not the case
    if not np.allclose(spectra.wave, skymodel.wave):
        message = "spectra and sky not on same wavelength grid"
        log.error(message)
        raise ValueError(message)

    spectra.flux -= skymodel.flux
    spectra.ivar = util.combine_ivar(spectra.ivar, skymodel.ivar)
    spectra.mask |= skymodel.mask

    log.info("done")
