"""
desispec.fiberflat
==================

Utility functions to compute a fiber flat correction and apply it
We try to keep all the (fits) io separated.
"""
from __future__ import absolute_import, division

import numpy as np
from desispec.resolution import Resolution
from desispec.linalg import cholesky_solve
from desispec.linalg import cholesky_solve_and_invert
from desispec.linalg import spline_fit
from desispec.maskbits import specmask
import scipy,scipy.sparse
import sys
from desispec.log import get_logger


def compute_fiberflat(frame, nsig_clipping=4.) :
    """Compute fiber flat by deriving an average spectrum and dividing all fiber data by this average.
    Input data are expected to be on the same wavelenght grid, with uncorrelated noise.
    They however do not have exactly the same resolution.

    Args:
        frame (desispec.Frame): input Frame object with attributes
            wave, flux, ivar, resolution_data
        nsig_clipping : [optional] sigma clipping value for outlier rejection

    Returns:
        desispec.FiberFlat object with attributes
            wave, fiberflat, ivar, mask, meanspec

    Notes:
    - we first iteratively :

       - compute a deconvolved mean spectrum
       - compute a fiber flat using the resolution convolved mean spectrum for each fiber
       - smooth the fiber flat along wavelength
       - clip outliers

    - then we compute a fiberflat at the native fiber resolution (not smoothed)

    - the routine returns the fiberflat, its inverse variance , mask, and the deconvolved mean spectrum

    - the fiberflat is the ratio data/mean , so this flat should be divided to the data

    NOTE THAT THIS CODE HAS NOT BEEN TESTED WITH ACTUAL FIBER TRANSMISSION VARIATIONS,
    OUTLIER PIXELS, DEAD COLUMNS ...
    """
    log=get_logger()
    log.info("starting")

    #
    # chi2 = sum_(fiber f) sum_(wavelenght i) w_fi ( D_fi - F_fi (R_f M)_i )
    #
    # where
    # w = inverse variance
    # D = flux data (at the resolution of the fiber)
    # F = smooth fiber flat
    # R = resolution data
    # M = mean deconvolved spectrum
    #
    # M = A^{-1} B
    # with
    # A_kl = sum_(fiber f) sum_(wavelenght i) w_fi F_fi^2 (R_fki R_fli)
    # B_k = sum_(fiber f) sum_(wavelenght i) w_fi D_fi F_fi R_fki
    #
    # defining R'_fi = sqrt(w_fi) F_fi R_fi
    # and      D'_fi = sqrt(w_fi) D_fi
    #
    # A = sum_(fiber f) R'_f R'_f^T
    # B = sum_(fiber f) R'_f D'_f
    # (it's faster that way, and we try to use sparse matrices as much as possible)
    #

    #- Shortcuts
    nwave=frame.nwave
    nfibers=frame.nspec
    wave = frame.wave.copy()  #- this will become part of output too
    flux = frame.flux
    ivar = frame.ivar


    # iterative fitting and clipping to get precise mean spectrum
    current_ivar=ivar.copy()


    smooth_fiberflat=np.ones((frame.flux.shape))
    chi2=np.zeros((flux.shape))


    sqrtwflat=np.sqrt(current_ivar)*smooth_fiberflat
    sqrtwflux=np.sqrt(current_ivar)*flux


    # test
    #nfibers=20
    nout_tot=0
    for iteration in range(20) :

        # fit mean spectrum
        A=scipy.sparse.lil_matrix((nwave,nwave)).tocsr()
        B=np.zeros((nwave))

        # diagonal sparse matrix with content = sqrt(ivar)*flat of a given fiber
        SD=scipy.sparse.lil_matrix((nwave,nwave))

        # loop on fiber to handle resolution
        for fiber in range(nfibers) :
            if fiber%10==0 :
                log.info("iter %d fiber %d"%(iteration,fiber))

            ### R = Resolution(resolution_data[fiber])
            R = frame.R[fiber]

            # diagonal sparse matrix with content = sqrt(ivar)*flat
            SD.setdiag(sqrtwflat[fiber])

            sqrtwflatR = SD*R # each row r of R is multiplied by sqrtwflat[r]

            A = A+(sqrtwflatR.T*sqrtwflatR).tocsr()
            B += sqrtwflatR.T*sqrtwflux[fiber]

        log.info("iter %d solving"%iteration)

        mean_spectrum=cholesky_solve(A.todense(),B)

        log.info("iter %d smoothing"%iteration)

        # fit smooth fiberflat and compute chi2
        smoothing_res=100. #A

        for fiber in range(nfibers) :

            #if fiber%10==0 :
            #    log.info("iter %d fiber %d (smoothing)"%(iteration,fiber))

            ### R = Resolution(resolution_data[fiber])
            R = frame.R[fiber]

            #M = np.array(np.dot(R.todense(),mean_spectrum)).flatten()
            M = R.dot(mean_spectrum)

            F = flux[fiber]/(M+(M==0))
            smooth_fiberflat[fiber]=spline_fit(wave,wave,F,smoothing_res,current_ivar[fiber]*(M!=0))
            chi2[fiber]=current_ivar[fiber]*(flux[fiber]-smooth_fiberflat[fiber]*M)**2

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
                sqrtwflat[worst_entry,i]=0
                sqrtwflux[worst_entry,i]=0
                nout_iter += 1

        else :
            # remove all of them at once
            bad=(chi2>nsig_clipping**2)
            current_ivar *= (bad==0)
            sqrtwflat *= (bad==0)
            sqrtwflux *= (bad==0)
            nout_iter += np.sum(bad)

        nout_tot += nout_iter

        sum_chi2=float(np.sum(chi2))
        ndf=int(np.sum(chi2>0)-nwave-nfibers*(nwave/smoothing_res))
        chi2pdf=0.
        if ndf>0 :
            chi2pdf=sum_chi2/ndf
        log.info("iter #%d chi2=%f ndf=%d chi2pdf=%f nout=%d"%(iteration,sum_chi2,ndf,chi2pdf,nout_iter))

        # normalize to get a mean fiberflat=1
        mean=np.mean(smooth_fiberflat,axis=0)
        smooth_fiberflat = smooth_fiberflat/mean
        mean_spectrum    = mean_spectrum*mean



        if nout_iter == 0 :
            break

    log.info("nout tot=%d"%nout_tot)

    # now use mean spectrum to compute flat field correction without any smoothing
    # because sharp feature can arise if dead columns

    fiberflat=np.ones((flux.shape))
    fiberflat_ivar=np.zeros((flux.shape))
    mask=np.zeros((flux.shape)).astype(long)  # SOMEONE CHECK THIS !

    fiberflat_mask=12 # place holder for actual mask bit when defined

    nsig_for_mask=4 # only mask out 4 sigma outliers

    for fiber in range(nfibers) :
        ### R = Resolution(resolution_data[fiber])
        R = frame.R[fiber]
        M = np.array(np.dot(R.todense(),mean_spectrum)).flatten()
        fiberflat[fiber] = (M!=0)*flux[fiber]/(M+(M==0)) + (M==0)
        fiberflat_ivar[fiber] = ivar[fiber]*M**2
        smooth_fiberflat=spline_fit(wave,wave,fiberflat[fiber],smoothing_res,current_ivar[fiber]*M**2*(M!=0))
        bad=np.where(fiberflat_ivar[fiber]*(fiberflat[fiber]-smooth_fiberflat)**2>nsig_for_mask**2)[0]
        if bad.size>0 :
            mask[fiber,bad] += fiberflat_mask

    return FiberFlat(wave, fiberflat, fiberflat_ivar, mask, mean_spectrum)    


def apply_fiberflat(frame, fiberflat):
### def apply_fiberflat(flux,ivar,wave,fiberflat,ffivar,ffmask,ffwave):
    """Apply fiberflat to frame.  Modifies frame.flux and frame.ivar
    
    Args:
        frame : `desispec.Frame` object
        fiberflat : `desispec.FiberFlat` object
        
    The frame is divided by the fiberflat, except where the fiberflat=0.

    frame.mask gets bit specmask.BADFIBERFLAT set where
      * fiberflat.fiberflat == 0
      * fiberflat.ivar == 0
      * fiberflat.mask != 0
    """
    log=get_logger()
    log.info("starting")

    # check same wavelength, die if not the case
    if not np.allclose(frame.wave, fiberflat.wave):
        message = "frame and fiberflat do not have the same wavelength arrays"
        log.critical(message)
        raise ValueError(message)

    """
     F'=F/C
     Var(F') = Var(F)/C**2 + F**2*(  d(1/C)/dC )**2*Var(C)
             = 1/(ivar(F)*C**2) + F**2*(1/C**2)**2*Var(C)
             = 1/(ivar(F)*C**2) + F**2*Var(C)/C**4
             = 1/(ivar(F)*C**2) + F**2/(ivar(C)*C**4)
    """
    #- shorthand
    ff = fiberflat
    sp = frame  #- sp=spectra for this frame
    
    ### sp.flux = sp.flux*(ff.fiberflat>0)/(ff.fiberflat+(ff.fiberflat==0))
    ii = np.where(ff.fiberflat > 0)
    sp.flux[ii] = sp.flux[ii] / ff.fiberflat[ii]

    sp.ivar=(sp.ivar>0)*(ff.ivar>0)*(ff.fiberflat>0)/( 1./((sp.ivar+(sp.ivar==0))*(ff.fiberflat**2+(ff.fiberflat==0))) + sp.flux**2/(ff.ivar*ff.fiberflat**4+(ff.ivar*ff.fiberflat==0)) )

    badff = (ff.fiberflat == 0.0) | (ff.ivar == 0) | (ff.mask != 0)
    sp.mask[badff] |= specmask.BADFIBERFLAT

    log.info("done")


class FiberFlat(object):
    def __init__(self, wave, fiberflat, ivar, mask=None, meanspec=None,
            header=None, fibers=None, spectrograph=0):
        """
        Creates a lightweight data wrapper for fiber flats

        Args:
            wave: 1D[nwave] wavelength in Angstroms
            fiberflat: 2D[nspec, nwave]
            ivar: 2D[nspec, nwave] inverse variance of fiberflat
            
        Optional inputs:
            mask: 2D[nspec, nwave] mask where 0=good; default ivar==0
            meanspec: 1D[nwave] mean deconvolved average flat lamp spectrum
            header: (optional) FITS header from HDU0
            fibers: (optional) fiber indices
            spectrograph: (optional) spectrograph number [0-9]       
        """
        if wave.ndim != 1:
            raise ValueError("wave should be 1D")

        if fiberflat.ndim != 2:
            raise ValueError("fiberflat should be 2D[nspec, nwave]")

        if ivar.ndim != 2:
            raise ValueError("ivar should be 2D")

        if fiberflat.shape != ivar.shape:
            raise ValueError("fiberflat and ivar must have the same shape")

        if mask is not None and mask.ndim != 2:
            raise ValueError("mask should be 2D")

        if meanspec is not None and meanspec.ndim != 1:
            raise ValueError("meanspec should be 1D")

        if mask is not None and fiberflat.shape != mask.shape:
            raise ValueError("fiberflat and mask must have the same shape")
        
        if meanspec is not None and wave.shape != meanspec.shape:
            raise ValueError("wrong size/shape for meanspec {}".format(meanspec.shape))
        
        if wave.shape[0] != fiberflat.shape[1]:
            raise ValueError("nwave mismatch between wave.shape[0] and flux.shape[1]")

        if mask is None:
            mask = (ivar == 0)

        if meanspec is None:
            meanspec = np.ones_like(wave)

        self.wave = wave
        self.fiberflat = fiberflat
        self.ivar = ivar
        self.mask = mask
        self.meanspec = meanspec

        self.nspec, self.nwave = self.fiberflat.shape
        self.header = header
        
        self.spectrograph = spectrograph
        if fibers is None:
            self.fibers = self.spectrograph + np.arange(self.nspec, dtype=int)
        else:
            if len(fibers) != self.nspec:
                raise ValueError("len(fibers) != nspec ({} != {})".format(len(fibers), self.nspec))
            self.fibers = fibers
            
    def __getitem__(self, index):
        """
        Return a subset of the spectra as a new FiberFlat object
        
        index can be anything that can index or slice a numpy array
        """
        #- convert index to 1d array to maintain dimentionality of sliced arrays
        if not isinstance(index, slice):
            index = np.atleast_1d(index)

        result = FiberFlat(self.wave, self.fiberflat[index], self.ivar[index],
                    self.mask[index], self.meanspec, header=self.header,
                    fibers=self.fibers[index], spectrograph=self.spectrograph)
        
        #- TODO:
        #- if we define fiber ranges in the fits headers, correct header
        
        return result
