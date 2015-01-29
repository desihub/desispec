"""
Utility functions to compute a fiber flat correction and apply it
We try to keep all the (fits) io separated.
"""


import numpy as np
from desispec.io.frame import resolution_data_to_sparse_matrix
from desispec.linalg import cholesky_solve
from desispec.linalg import cholesky_solve_and_invert
from desispec.linalg import spline_fit
import scipy,scipy.sparse
import sys

def compute_fiberflat(wave,flux,ivar,resolution_data,nsig_clipping=4.) :
    
    """ 
    compute fiber flat by deriving an average spectrum and dividing all fiber data by this average.
    input data are expected to be on the same wavelenght grid, with uncorrelated noise.
    they however do not have exactly the same resolution.
    
    args:
        wave : 1D wavelength grid in Angstroms
        flux : 2D flux[nspec, nwave] density
        ivar : 2D inverse variance of flux
        resolution_data : 3D[nspec, ndiag, nwave] ...
        nsig_clipping : [optional] sigma clipping value for outlier rejection
        
    returns tuple (fiberflat, ivar, meanspec):
        fiberflat : 2D[nwave, nflux] fiberflat (divide or multiply?)
        ivar : inverse variance of that fiberflat
        meanspec : deconvolved mean spectrum

    - we first iteratively :
       - compute a deconvolved mean spectrum
       - compute a fiber flat using the resolution convolved mean spectrum for each fiber
       - smooth the fiber flat along wavelength 
       - clip outliers

    - then we compute a fiberflat at the native fiber resolution (not smoothed)
    
    - the routine returns the fiberflat, its inverse variance , and the deconvolved mean spectrum


    NOTE THAT THIS CODE HAS NOT BEEN TESTED WITH ACTUAL FIBER TRANSMISSION VARIATIONS,
    OUTLIER PIXELS, DEAD COLUMNS ...
    
    """
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
    # nfibers=20
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
                print "fiber",fiber
            R = resolution_data_to_sparse_matrix(resolution_data,fiber)
            
            # diagonal sparse matrix with content = sqrt(ivar)*flat
            SD.setdiag(sqrtwflat[fiber])
                        
            sqrtwflatR = SD*R # each row r of R is multiplied by sqrtwflat[r] 
            
            A = A+(sqrtwflatR.T*sqrtwflatR).tocsr()
            B += sqrtwflatR.T*sqrtwflux[fiber]
        
        print "solving"
        mean_spectrum=cholesky_solve(A.todense(),B)
        
        print "smoothing"
        # fit smooth fiberflat and compute chi2
        smoothing_res=100. #A
        
        for fiber in range(nfibers) :
            if fiber%10==0 :
                print "fiber",fiber
            R = resolution_data_to_sparse_matrix(resolution_data,fiber)
            
            #M = np.array(np.dot(R.todense(),mean_spectrum)).flatten()
            M = R.dot(mean_spectrum)
                        
            F = flux[fiber]/(M+(M==0))
            smooth_fiberflat[fiber]=spline_fit(wave,wave,F,smoothing_res,current_ivar[fiber]*(M!=0))
            chi2[fiber]=current_ivar[fiber]*(flux[fiber]-smooth_fiberflat[fiber]*M)**2
        
        print "rejecting"
        
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
        print "iter #%d chi2=%f ndf=%d chi2pdf=%f nout=%d"%(iteration,sum_chi2,ndf,chi2pdf,nout_iter)

        # normalize to get a mean fiberflat=1
        mean=np.mean(smooth_fiberflat,axis=0)
        smooth_fiberflat = smooth_fiberflat/mean
        mean_spectrum    = mean_spectrum*mean
        


        if nout_iter == 0 :
            break
    
    print "nout tot=",nout_tot

    # now use mean spectrum to compute flat field correction without any smoothing
    # because sharp feature can arise if dead columns
    
    fiberflat=np.ones((flux.shape))
    fiberflat_ivar=np.zeros((flux.shape))
    
    for fiber in range(nfibers) :
        R = resolution_data_to_sparse_matrix(resolution_data,fiber)
        M = np.array(np.dot(R.todense(),mean_spectrum)).flatten()
        fiberflat[fiber] = (M!=0)*flux[fiber]/(M+(M==0)) + (M==0)
        fiberflat_ivar[fiber] = ivar[fiber]*M**2

    return fiberflat,fiberflat_ivar,mean_spectrum

    


