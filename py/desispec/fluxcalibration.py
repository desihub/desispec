"""
Utility functions to compute a fiber flat correction and apply it
We try to keep all the (fits) io separated.
"""


import numpy as np
from desispec.io.frame import resolution_data_to_sparse_matrix
from desispec.linalg import cholesky_solve
from desispec.linalg import cholesky_solve_and_invert
from desispec.linalg import spline_fit
from desispec.interpolation import resample_flux
from desispec.log import get_logger

import scipy,scipy.sparse
import sys
#debug
import pylab

def compute_flux_calibration(wave,flux,ivar,resolution_data,input_model_wave,input_model_flux,nsig_clipping=4.) :
    
    """ 
    compute average frame throughtput based on data (wave,flux,ivar,resolution_data)
    and spectro-photometrically calibrated stellar models (model_wave,model_flux)
    wave and model_wave are not necessarily on the same grid
    
    input flux and model fiber indices have to match
    
    - we first resample the model on the input flux wave grid
    - then convolve it to the data resolution (the input wave grid is supposed finer than the spectral resolution)
    - then iteratively 
       - fit the mean throughput (deconvolved, this is needed because of sharp atmospheric absorption lines)
       - compute broad band correction to fibers (to correct for small mis-alignement for instance)
       - performe an outlier rejection
    """
    
    log=get_logger()
    log.info("starting")

    nwave=wave.size
    nfibers=flux.shape[0]
    

    # resample model to data grid and convolve by resolution
    model_flux=np.zeros(flux.shape)
    for fiber in range(model_flux.shape[0]) :
        model_flux[fiber]=resample_flux(wave,input_model_wave,input_model_flux[fiber],left=0.,right=0.)
        
        # debug
        # pylab.plot(input_model_wave,input_model_flux[fiber])
        # pylab.plot(wave,model_flux[fiber],c="g")

        R = resolution_data_to_sparse_matrix(resolution_data,fiber)
        model_flux[fiber]=R.dot(model_flux[fiber])
        
        # debug
        # pylab.plot(wave,model_flux[fiber],c="r")
        # pylab.show()
    

    

    # iterative fitting and clipping to get precise mean spectrum
    current_ivar=ivar.copy()
    
    
    smooth_fiber_correction=np.ones((flux.shape))
    chi2=np.zeros((flux.shape))
    

    sqrtwmodel=np.sqrt(current_ivar)*model_flux
    sqrtwflux=np.sqrt(current_ivar)*flux
    

    # test
    # nfibers=20
    nout_tot=0
    for iteration in range(20) :

        # fit mean calibration
        A=scipy.sparse.lil_matrix((nwave,nwave)).tocsr()
        B=np.zeros((nwave))
        
        # diagonal sparse matrix with content = sqrt(ivar)*flat of a given fiber
        SD=scipy.sparse.lil_matrix((nwave,nwave))
        
        # loop on fiber to handle resolution
        for fiber in range(nfibers) :
            if fiber%10==0 :
                log.info("iter %d fiber %d"%(iteration,fiber))
            R = resolution_data_to_sparse_matrix(resolution_data,fiber)
            
            # diagonal sparse matrix with content = sqrt(ivar)*flat
            SD.setdiag(sqrtwmodel[fiber])
                        
            sqrtwmodelR = SD*R # each row r of R is multiplied by sqrtwmodel[r] 
            
            A = A+(sqrtwmodelR.T*sqrtwmodelR).tocsr()
            B += sqrtwmodelR.T*sqrtwflux[fiber]
        
        log.info("iter %d solving"%iteration)
        calibration=cholesky_solve(A.todense(),B)
        #pylab.plot(wave,calibration)
        #pylab.show()
        #sys.exit(12)
        
        log.info("iter %d fit smooth correction per fiber"%iteration)
        # fit smooth fiberflat and compute chi2
        smoothing_res=1000. #A
        
        for fiber in range(nfibers) :
            if fiber%10==0 :
                log.info("iter %d fiber %d(smooth)"%(iteration,fiber))
            
            R = resolution_data_to_sparse_matrix(resolution_data,fiber)
            
            #M = np.array(np.dot(R.todense(),mean_spectrum)).flatten()
            M = R.dot(calibration)*model_flux[fiber]
            
            #debug
            #pylab.plot(wave,flux[fiber],c="b")
            #pylab.plot(wave,M,c="r")
            #pylab.show()
            #continue
        
            F = flux[fiber]/(M+(M==0))
            smooth_fiber_correction[fiber]=spline_fit(wave,wave,F,smoothing_res,current_ivar[fiber]*(M!=0))
            chi2[fiber]=current_ivar[fiber]*(flux[fiber]-smooth_fiber_correction[fiber]*M)**2
            
            #pylab.plot(wave,F)
            #pylab.plot(wave,smooth_fiber_correction[fiber])
        
        
        #pylab.show()
        #sys.exit(12)


        log.info("iter %d rejecting"%iteration)
        
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
                sqrtwmodel[worst_entry,i]=0
                sqrtwflux[worst_entry,i]=0
                nout_iter += 1
                
        else :
            # remove all of them at once
            bad=(chi2>nsig_clipping**2)
            current_ivar *= (bad==0)
            sqrtwmodel *= (bad==0)
            sqrtwflux *= (bad==0)
            nout_iter += np.sum(bad)
        
        nout_tot += nout_iter

        sum_chi2=float(np.sum(chi2))
        ndf=int(np.sum(chi2>0)-nwave-nfibers*(nwave/smoothing_res))
        chi2pdf=0.
        if ndf>0 :
            chi2pdf=sum_chi2/ndf
        
        # normalize to get a mean fiberflat=1
        mean=np.mean(smooth_fiber_correction,axis=0)
        smooth_fiber_correction = smooth_fiber_correction/mean
        calibration *= mean
        
        log.info("iter #%d chi2=%f ndf=%d chi2pdf=%f nout=%d mean=%f"%(iteration,sum_chi2,ndf,chi2pdf,nout_iter,np.mean(mean)))
        


        if nout_iter == 0 :
            break
    
    log.info("nout tot=%d"%nout_tot)
    
    # solve once again to get deconvolved variance
    #calibration,calibcovar=cholesky_solve_and_invert(A.todense(),B)
    calibcovar=np.linalg.inv(A.todense())
    calibvar=np.diagonal(calibcovar)
    print "mean(var)=",np.mean(calibvar)

    
    
    calibvar=np.array(np.diagonal(calibcovar))
    # apply the mean (as in the iterative loop)
    calibvar *= mean**2
    calibivar=(calibvar>0)/(calibvar+(calibvar==0))
    
    # we also want to save the convolved calibration and calibration variance
    # first compute average resolution
    mean_res_data=np.mean(resolution_data,axis=0)
    R = resolution_data_to_sparse_matrix(mean_res_data,0)
    # compute convolved calib and ivar
    ccalibration=R.dot(calibration)
    ccalibcovar=R.dot(calibcovar).dot(R.T.todense())
    ccalibvar=np.array(np.diagonal(ccalibcovar))
    # apply the mean (as in the iterative loop)
    ccalibvar *= mean**2
    ccalibivar=(ccalibvar>0)/(ccalibvar+(ccalibvar==0))
    
    

    # need to do better here 
    mask=(calibvar>0).astype(long)  # SOMEONE CHECK THIS !
    
    return calibration, calibivar, mask, ccalibration, ccalibivar

def apply_flux_calibration(flux,ivar,resolution_data=resol,wave,calibration,civar,cmask,cwave) :
    
    log=get_logger()
    log.info("starting")

    # check same wavelength, die if not the case
    mval=np.max(np.abs(wave-skywave))
    if mval > 0.00001 :
        log.error("not same wavelength (should raise an error instead)")
        sys.exit(12)
    
    nwave=wave.size
    nfibers=flux.shape[0]

    for fiber in range(nfibers) :

        R = resolution_data_to_sparse_matrix(resolution_data,fiber)
        C = R.dot(calibration)
    
        """
        F'=F/C
        Var(F') = Var(F)/C**2 + F**2*(  d(1/C)/dC )**2*Var(C)
        = 1/(ivar(F)*C**2) + F**2*(1/C**2)**2*Var(C)
        = 1/(ivar(F)*C**2) + F**2*Var(C)/C**4
        = 1/(ivar(F)*C**2) + F**2/(ivar(C)*C**4)
        """
        
        flux=flux*(C>0)/(C+(C==0))
        ivar=(ivar>0)*(civar>0)*(C>0)/(   1./((ivar+(ivar==0))*(C**2+(C==0))) + flux**2/(civar*C**4+(civar*C==0))   )
    
    

