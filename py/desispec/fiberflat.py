"""
desispec.fiberflat
==================

Utility functions to compute a fiber flat correction and apply it
We try to keep all the (fits) io separated.
"""
from __future__ import absolute_import, division

import numpy as np
# from desispec.io.frame import resolution_data_to_sparse_matrix
from desispec.resolution import Resolution
from desispec.linalg import cholesky_solve
from desispec.linalg import cholesky_solve_and_invert
from desispec.linalg import spline_fit
import scipy,scipy.sparse
import sys
from desispec.log import get_logger


def compute_fiberflat(wave,flux,ivar,resolution_data,nsig_clipping=4.) :
    """Compute fiber flat by deriving an average spectrum and dividing all fiber data by this average.
    Input data are expected to be on the same wavelenght grid, with uncorrelated noise.
    They however do not have exactly the same resolution.

    args:
        wave : 1D wavelength grid in Angstroms
        flux : 2D flux[nspec, nwave] density
        ivar : 2D inverse variance of flux
        resolution_data : 3D[nspec, ndiag, nwave] ...
        nsig_clipping : [optional] sigma clipping value for outlier rejection

    returns tuple (fiberflat, ivar, mask, meanspec):
        fiberflat : 2D[nwave, nflux] fiberflat (data have to be divided by this to be flatfielded)
        ivar : inverse variance of that fiberflat
        mask : 0=ok >0 if problems
        meanspec : deconvolved mean spectrum

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


    nwave=wave.size
    nfibers=flux.shape[0]



    # iterative fitting and clipping to get precise mean spectrum
    current_ivar=ivar.copy()


    smooth_fiberflat=np.ones((flux.shape))
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

            ### R = resolution_data_to_sparse_matrix(resolution_data,fiber)
            R = Resolution(resolution_data[fiber])

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

            ### R = resolution_data_to_sparse_matrix(resolution_data,fiber)
            R = Resolution(resolution_data[fiber])

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
        ### R = resolution_data_to_sparse_matrix(resolution_data,fiber)
        R = Resolution(resolution_data[fiber])
        M = np.array(np.dot(R.todense(),mean_spectrum)).flatten()
        fiberflat[fiber] = (M!=0)*flux[fiber]/(M+(M==0)) + (M==0)
        fiberflat_ivar[fiber] = ivar[fiber]*M**2
        smooth_fiberflat=spline_fit(wave,wave,fiberflat[fiber],smoothing_res,current_ivar[fiber]*M**2*(M!=0))
        bad=np.where(fiberflat_ivar[fiber]*(fiberflat[fiber]-smooth_fiberflat)**2>nsig_for_mask**2)[0]
        if bad.size>0 :
            mask[fiber,bad] += fiberflat_mask

    return fiberflat,fiberflat_ivar,mask,mean_spectrum



def apply_fiberflat(flux,ivar,wave,fiberflat,ffivar,ffmask,ffwave):
    """No documentation yet.
    """
    log=get_logger()
    log.info("starting")

    # check same wavelength, die if not the case
    mval=np.max(np.abs(wave-ffwave))
    if mval > 0.00001 :
        log.critical("error in apply_fiberflat, not same wavelength (should raise an error instead)")
        sys.exit(12)

    """
     F'=F/C
     Var(F') = Var(F)/C**2 + F**2*(  d(1/C)/dC )**2*Var(C)
             = 1/(ivar(F)*C**2) + F**2*(1/C**2)**2*Var(C)
             = 1/(ivar(F)*C**2) + F**2*Var(C)/C**4
             = 1/(ivar(F)*C**2) + F**2/(ivar(C)*C**4)
    """

    flux=flux*(fiberflat>0)/(fiberflat+(fiberflat==0))
    ivar=(ivar>0)*(ffivar>0)*(fiberflat>0)/(   1./((ivar+(ivar==0))*(fiberflat**2+(fiberflat==0))) + flux**2/(ffivar*fiberflat**4+(ffivar*fiberflat==0))   )


    log.info("done")
