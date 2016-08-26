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
from desispec import util
import scipy,scipy.sparse
import sys
from desispec.log import get_logger
import math


def compute_fiberflat(frame, nsig_clipping=4., accuracy=5.e-4, minval=0.1, maxval=10.) :
    """Compute fiber flat by deriving an average spectrum and dividing all fiber data by this average.
    Input data are expected to be on the same wavelength grid, with uncorrelated noise.
    They however do not have exactly the same resolution.

    Args:
        frame (desispec.Frame): input Frame object with attributes
            wave, flux, ivar, resolution_data
        nsig_clipping : [optional] sigma clipping value for outlier rejection
        accuracy : [optional] accuracy of fiberflat (end test for the iterative loop)
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
    ivar = frame.ivar*(frame.mask==0)
    
    
    
    # iterative fitting and clipping to get precise mean spectrum


   

    # we first need to iterate to converge on a solution of mean spectrum
    # and smooth fiber flat. several interations are needed when
    # throughput AND resolution vary from fiber to fiber.
    # the end test is that the fiber flat has varied by less than accuracy
    # of previous iteration for all wavelength
    # we also have a max. number of iterations for this code
    max_iterations = 100
    
    nout_tot=0
    chi2pdf = 0.
    
    smooth_fiberflat=np.ones((frame.flux.shape))
    previous_smooth_fiberflat=smooth_fiberflat.copy()
    
    chi2=np.zeros((flux.shape))


    # 1st pass is median for spectrum, flat field without resolution
    # outlier rejection
    
    for iteration in range(max_iterations) :
        
        # use median for spectrum
        mean_spectrum=np.zeros((flux.shape[1]))
        for i in range(flux.shape[1]) :
            ok=np.where(ivar[:,i]>0)[0]
            if ok.size > 0 :
                mean_spectrum[i]=np.median(flux[ok,i])
                
        # max pixels far from mean spectrum.
        #log.info("mask pixels with difference smaller than %f or larger than %f of mean")
        nout_iter=0
        for fiber in range(nfibers) :
            bad=np.where((ivar[fiber]>0)&((flux[fiber]>maxval*mean_spectrum)|(flux[fiber]<minval*mean_spectrum)))[0]
        if bad.size>100 :
            log.warning("masking fiber %d because of bad flat field with %d bad pixels"%(fiber,bad.size))
            ivar[fiber]=0.                
        if bad.size>0 :
            log.warning("masking %d bad pixels for fiber %d"%(bad.size,fiber))
            ivar[fiber,bad]=0.
        nout_iter += bad.size
        
        # fit smooth fiberflat and compute chi2
        smoothing_res=100. #A
        
        for fiber in range(nfibers) :
            
            if np.sum(ivar[fiber]>0)==0 :
                continue

            F = np.ones((flux.shape[1]))
            ok=np.where((mean_spectrum!=0)&(ivar[fiber]>0))[0]
            F[ok] = flux[fiber,ok]/mean_spectrum[ok]
            smooth_fiberflat[fiber]=spline_fit(wave,wave[ok],F[ok],smoothing_res,ivar[fiber,ok])
            
        
        # normalize to get a mean fiberflat=1
        mean=np.mean(smooth_fiberflat,axis=0)
        ok=np.where(mean!=0)[0]
        for fiber in range(nfibers) :
            smooth_fiberflat[fiber,ok] = smooth_fiberflat[fiber,ok]/mean[ok]
        mean_spectrum *= mean
                
        
        
        # this is the max difference between two iterations
        max_diff=np.max(np.abs(smooth_fiberflat-previous_smooth_fiberflat)*(ivar>0.)) 
        previous_smooth_fiberflat=smooth_fiberflat.copy()
        
        # we don't start the rejection tests until we have converged on this
        if max_diff>0.01 :
            log.info("1st pass, max diff. = %g > 0.01 , continue iterating before outlier rejection"%(max_diff))
            continue
                    

        chi2=ivar*(flux-smooth_fiberflat*mean_spectrum)**2
        
        if True :  
            nsig_clipping_for_this_pass = nsig_clipping
            
            # not more than 5 pixels per fiber at a time
            for fiber in range(nfibers) :
                for loop in range(max_iterations) :
                    bad=np.where(chi2[fiber]>nsig_clipping_for_this_pass**2)[0]
                    if bad.size>0 :                
                        if bad.size>5 : # not more than 5 pixels at a time
                            ii=np.argsort(chi2[fiber,bad])
                            bad=bad[ii[-5:]]
                        ivar[fiber,bad] = 0
                        nout_iter += bad.size
                        ok=np.where((mean_spectrum!=0)&(ivar[fiber]>0))[0]
                        F[ok] = flux[fiber,ok]/mean_spectrum[ok]
                        smooth_fiberflat[fiber]=spline_fit(wave,wave[ok],F[ok],smoothing_res,ivar[fiber,ok])
                        chi2[fiber]=ivar[fiber]*(flux[fiber]-smooth_fiberflat[fiber]*mean_spectrum)**2
                    else :
                        break
        
            nout_tot += nout_iter

            sum_chi2=float(np.sum(chi2))
            ndf=int(np.sum(chi2>0)-nwave-nfibers*(nwave/smoothing_res))
            chi2pdf=0.
            if ndf>0 :
                chi2pdf=sum_chi2/ndf
            log.info("1st pass iter #%d chi2=%f ndf=%d chi2pdf=%f nout=%d (nsig=%f)"%(iteration,sum_chi2,ndf,chi2pdf,nout_iter,nsig_clipping_for_this_pass))

        
        if max_diff>accuracy :
            log.info("1st pass iter #%d max diff. = %g > requirement = %g , continue iterating"%(iteration,max_diff,accuracy))
            continue
    
        if nout_iter == 0 :
            break

    log.info("after 1st pass : nout = %d/%d"%(np.sum(ivar==0),np.size(ivar.flatten())))
    
    # 2nd pass is full solution including deconvolved spectrum, no outlier rejection
    for iteration in range(max_iterations) : 
        
        log.info("2nd pass, iter %d : mean deconvolved spectrum"%iteration)
        
        # fit mean spectrum
        A=scipy.sparse.lil_matrix((nwave,nwave)).tocsr()
        B=np.zeros((nwave))

        # diagonal sparse matrix with content = sqrt(ivar)*flat of a given fiber
        SD=scipy.sparse.lil_matrix((nwave,nwave))

        # this is to go a bit faster
        sqrtwflat=np.sqrt(ivar)*smooth_fiberflat
        
        # loop on fiber to handle resolution (this is long)
        for fiber in range(nfibers) :
            if fiber%10==0 :
                log.info("2nd pass, filling matrix, iter %d fiber %d"%(iteration,fiber))
                
            ### R = Resolution(resolution_data[fiber])
            R = frame.R[fiber]                
            SD.setdiag(sqrtwflat[fiber])

            sqrtwflatR = SD*R # each row r of R is multiplied by sqrtwflat[r]
                
            A = A+(sqrtwflatR.T*sqrtwflatR).tocsr()
            B += sqrtwflatR.T.dot(np.sqrt(ivar[fiber])*flux[fiber])
            
        mean_spectrum=cholesky_solve(A.todense(),B)
            
            
        # fit smooth fiberflat
        smoothing_res=100. #A

        for fiber in range(nfibers) :

            if np.sum(ivar[fiber]>0)==0 :
                continue
            
            ### R = Resolution(resolution_data[fiber])
            R = frame.R[fiber]
            
            M = R.dot(mean_spectrum)            
            ok=np.where(M!=0)[0]
            smooth_fiberflat[fiber]=spline_fit(wave,wave[ok],flux[fiber,ok]/M[ok],smoothing_res,ivar[fiber,ok])
        
        # normalize to get a mean fiberflat=1
        mean=np.mean(smooth_fiberflat,axis=0)
        ok=np.where(mean!=0)[0]
        smooth_fiberflat[:,ok] /= mean[ok]
        mean_spectrum *= mean
        
        chi2=ivar*(flux-smooth_fiberflat*mean_spectrum)**2
        
        # this is the max difference between two iterations
        max_diff=np.max(np.abs(smooth_fiberflat-previous_smooth_fiberflat)*(ivar>0.))
        previous_smooth_fiberflat=smooth_fiberflat.copy()
        
        sum_chi2=float(np.sum(chi2))
        ndf=int(np.sum(chi2>0)-nwave-nfibers*(nwave/smoothing_res))
        chi2pdf=0.
        if ndf>0 :
            chi2pdf=sum_chi2/ndf
        log.info("2nd pass, iter %d, chi2=%f ndf=%d chi2pdf=%f"%(iteration,sum_chi2,ndf,chi2pdf))
        
        if max_diff<accuracy :
            break
        
        log.info("2nd pass, iter %d, max diff. = %g > requirement = %g, continue iterating"%(iteration,max_diff,accuracy))
        

    log.info("Total number of masked pixels=%d"%nout_tot)

    log.info("3rd pass, final computation of fiber flat")

    # now use mean spectrum to compute flat field correction without any smoothing
    # because sharp feature can arise if dead columns

    fiberflat=np.ones((flux.shape))
    fiberflat_ivar=np.zeros((flux.shape))
    mask=np.zeros((flux.shape), dtype='uint32')
    
    # reset ivar
    ivar=frame.ivar
    
    fiberflat_mask=12 # place holder for actual mask bit when defined
    
    nsig_for_mask=nsig_clipping # only mask out N sigma outliers

    for fiber in range(nfibers) :
        
        if np.sum(ivar[fiber]>0)==0 :
            continue

        ### R = Resolution(resolution_data[fiber])
        R = frame.R[fiber]
        M = np.array(np.dot(R.todense(),mean_spectrum)).flatten()
        fiberflat[fiber] = (M!=0)*flux[fiber]/(M+(M==0)) + (M==0)
        fiberflat_ivar[fiber] = ivar[fiber]*M**2
        nbad_tot=0
        iteration=0
        while iteration<500 :
            smooth_fiberflat=spline_fit(wave,wave,fiberflat[fiber],smoothing_res,fiberflat_ivar[fiber])
            chi2=fiberflat_ivar[fiber]*(fiberflat[fiber]-smooth_fiberflat)**2
            bad=np.where(chi2>nsig_for_mask**2)[0]
            if bad.size>0 :
                
                if bad.size>5 : # not more than 5 pixels at a time
                    ii=np.argsort(chi2[bad])
                    bad=bad[ii[-5:]]
                
                mask[fiber,bad] += fiberflat_mask
                fiberflat_ivar[fiber,bad] = 0.
                nbad_tot += bad.size
            else :
                break
            iteration += 1
        # replace bad by smooth fiber flat
        bad=np.where((mask[fiber]>0)|(fiberflat_ivar[fiber]==0)|(fiberflat[fiber]<minval)|(fiberflat[fiber]>maxval))[0]
        if bad.size>0 :

            fiberflat_ivar[fiber,bad] = 0

            # find max length of segment with bad pix
            length=0
            for i in range(bad.size) :
                ib=bad[i]
                ilength=1
                tmp=ib
                for jb in bad[i+1:] :
                    if jb==tmp+1 :
                        ilength +=1
                        tmp=jb
                    else :
                        break
                length=max(length,ilength)
            if length>10 :
                log.info("3rd pass : fiber #%d has a max length of bad pixels=%d"%(fiber,length))
            smoothing_res=float(max(100,2*length))
            x=np.arange(wave.size)
            
            ok=np.where(fiberflat_ivar[fiber]>0)[0]
            smooth_fiberflat=spline_fit(x,x[ok],fiberflat[fiber,ok],smoothing_res,fiberflat_ivar[fiber,ok])
            fiberflat[fiber,bad] = smooth_fiberflat[bad]
                    
        if nbad_tot>0 :
            log.info("3rd pass : fiber #%d masked pixels = %d (%d iterations)"%(fiber,nbad_tot,iteration))
    
    # set median flat to 1
    log.info("set median fiberflat to 1")
    
    mean=np.ones((flux.shape[1]))
    for i in range(flux.shape[1]) :
        ok=np.where((mask[:,i]==0)&(ivar[:,i]>0))[0]
        if ok.size > 0 :
            mean[i] = np.median(fiberflat[ok,i])
    ok=np.where(mean!=0)[0]
    for fiber in range(nfibers) :
        fiberflat[fiber,ok] /= mean[ok]

    log.info("done fiberflat")

    return FiberFlat(wave, fiberflat, fiberflat_ivar, mask, mean_spectrum,
                     chi2pdf=chi2pdf)


def apply_fiberflat(frame, fiberflat):
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
    
    #- update sp.ivar first since it depends upon the original sp.flux
    sp.ivar=(sp.ivar>0)*(ff.ivar>0)*(ff.fiberflat>0)/( 1./((sp.ivar+(sp.ivar==0))*(ff.fiberflat**2+(ff.fiberflat==0))) + sp.flux**2/(ff.ivar*ff.fiberflat**4+(ff.ivar*ff.fiberflat==0)) )

    #- Then update sp.flux, taking care not to divide by 0
    ii = np.where(ff.fiberflat > 0)
    sp.flux[ii] = sp.flux[ii] / ff.fiberflat[ii]

    badff = (ff.fiberflat == 0.0) | (ff.ivar == 0) | (ff.mask != 0)
    sp.mask[badff] |= specmask.BADFIBERFLAT

    log.info("done")


class FiberFlat(object):
    def __init__(self, wave, fiberflat, ivar, mask=None, meanspec=None,
            chi2pdf=None, header=None, fibers=None, spectrograph=0):
        """
        Creates a lightweight data wrapper for fiber flats

        Args:
            wave: 1D[nwave] wavelength in Angstroms
            fiberflat: 2D[nspec, nwave]
            ivar: 2D[nspec, nwave] inverse variance of fiberflat
            
        Optional inputs:
            mask: 2D[nspec, nwave] mask where 0=good; default ivar==0; 32-bit
            meanspec: (optional) 1D[nwave] mean deconvolved average flat lamp spectrum
            chi2pdf: (optional) Normalized chi^2 for fit to mean spectrum
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
        self.mask = util.mask32(mask)
        self.meanspec = meanspec

        self.nspec, self.nwave = self.fiberflat.shape
        self.header = header

        if chi2pdf is not None:
            self.chi2pdf = chi2pdf
        else:
            try:
                self.chi2pdf = header['chi2pdf']
            except (KeyError, TypeError):
                self.chi2pdf = None

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

    def __repr__(self):
        """ Print formatting
        """
        return ('{:s}: nspec={:d}, spectrograph={:d}'.format(
                self.__class__.__name__, self.nspec, self.spectrograph))


def qa_fiberflat(param, frame, fiberflat):
    """ Calculate QA on FiberFlat object

    Args:
        param: dict of QA parameters
        frame: Frame
        fiberflat: FiberFlat

    Returns:
        qadict: dict of QA outputs
          Need to record simple Python objects for yaml (str, float, int)
    """
    log = get_logger()

    # Output dict
    qadict = {}

    # Check amplitude of the meanspectrum
    qadict['MAX_MEANSPEC'] = float(np.max(fiberflat.meanspec))
    if qadict['MAX_MEANSPEC'] < 100000:
        log.warning("Low counts in meanspec = {:g}".format(qadict['MAX_MEANSPEC']))

    # Record chi2pdf
    try:
        qadict['CHI2PDF'] = float(fiberflat.chi2pdf)
    except TypeError:
        qadict['CHI2PDF'] = 0.

    # N mask
    qadict['N_MASK'] = int(np.sum(fiberflat.mask > 0))
    if qadict['N_MASK'] > param['MAX_N_MASK']:  # Arbitrary
        log.warn("High rejection rate: {:d}".format(qadict['N_MASK']))

    # Scale (search for low/high throughput)
    gdp = fiberflat.mask == 0
    rtio = frame.flux / np.outer(np.ones(fiberflat.nspec),fiberflat.meanspec)
    scale = np.median(rtio*gdp,axis=1)
    MAX_SCALE_OFF = float(np.max(np.abs(scale-1.)))
    fiber = int(np.argmax(np.abs(scale-1.)))
    qadict['MAX_SCALE_OFF'] = [MAX_SCALE_OFF, fiber]
    if qadict['MAX_SCALE_OFF'][0] > param['MAX_SCALE_OFF']:
        log.warn("Discrepant flux in fiberflat: {:g}, {:d}".format(
                qadict['MAX_SCALE_OFF'][0], qadict['MAX_SCALE_OFF'][1]))

    # Offset in fiberflat
    qadict['MAX_OFF'] = float(np.max(np.abs(fiberflat.fiberflat-1.)))
    if qadict['MAX_OFF'] > param['MAX_OFF']:
        log.warn("Large offset in fiberflat: {:g}".format(qadict['MAX_OFF']))

    # Offset in mean of fiberflat
    mean = np.mean(fiberflat.fiberflat*gdp,axis=1)
    fiber = int(np.argmax(np.abs(mean-1.)))
    qadict['MAX_MEAN_OFF'] = [float(np.max(np.abs(mean-1.))), fiber]
    if qadict['MAX_MEAN_OFF'][0] > param['MAX_MEAN_OFF']:
        log.warn("Discrepant mean in fiberflat: {:g}, {:d}".format(
                qadict['MAX_MEAN_OFF'][0], qadict['MAX_MEAN_OFF'][1]))

    # RMS in individual fibers
    rms = np.std(gdp*(fiberflat.fiberflat-
                      np.outer(mean, np.ones(fiberflat.nwave))),axis=1)
    fiber = int(np.argmax(rms))
    qadict['MAX_RMS'] = [float(np.max(rms)), fiber]
    if qadict['MAX_RMS'][0] > param['MAX_RMS']:
        log.warn("Large RMS in fiberflat: {:g}, {:d}".format(
                qadict['MAX_RMS'][0], qadict['MAX_RMS'][1]))

    # Return
    return qadict

