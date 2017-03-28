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
from desiutil.log import get_logger
import math


def compute_fiberflat(frame, nsig_clipping=10., accuracy=5.e-4, minval=0.1, maxval=10.,max_iterations=100,smoothing_res=20.,max_bad=100,max_rej_it=5,min_sn=0,diag_epsilon=1e-3) :
    """Compute fiber flat by deriving an average spectrum and dividing all fiber data by this average.
    Input data are expected to be on the same wavelength grid, with uncorrelated noise.
    They however do not have exactly the same resolution.

    Args:
        frame (desispec.Frame): input Frame object with attributes
            wave, flux, ivar, resolution_data
        nsig_clipping : [optional] sigma clipping value for outlier rejection
        accuracy : [optional] accuracy of fiberflat (end test for the iterative loop)
        minval: [optional] mask pixels with flux < minval * median fiberflat.
        maxval: [optional] mask pixels with flux > maxval * median fiberflat.
        max_iterations: [optional] maximum number of iterations
        smoothing_res: [optional] spacing between spline fit nodes for smoothing the fiberflat
        max_bad: [optional] mask entire fiber if more than max_bad-1 initially unmasked pixels are masked during the iterations
        max_rej_it: [optional] reject at most the max_rej_it worst pixels in each iteration
        min_sn: [optional] mask portions with signal to noise less than min_sn
        diag_epsilon: [optional] size of the regularization term in the deconvolution


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
    flux = frame.flux.copy()
    ivar = frame.ivar*(frame.mask==0)



    # iterative fitting and clipping to get precise mean spectrum




    # we first need to iterate to converge on a solution of mean spectrum
    # and smooth fiber flat. several interations are needed when
    # throughput AND resolution vary from fiber to fiber.
    # the end test is that the fiber flat has varied by less than accuracy
    # of previous iteration for all wavelength
    # we also have a max. number of iterations for this code

    nout_tot=0
    chi2pdf = 0.

    smooth_fiberflat=np.ones((flux.shape))

    chi2=np.zeros((flux.shape))

    ## mask low sn portions
    w = flux*np.sqrt(ivar)<min_sn
    ivar[w]=0

    ## 0th pass: reject pixels according to minval and maxval
    mean_spectrum = np.zeros(flux.shape[1])
    nbad=np.zeros(nfibers,dtype=int)
    for iteration in range(max_iterations):
        for i in range(flux.shape[1]):
            w = ivar[:,i]>0
            if w.sum()>0:
                mean_spectrum[i] = np.median(flux[w,i])

        nbad_it=0
        for fib in range(nfibers):
            w = ((flux[fib,:]<minval*mean_spectrum) | (flux[fib,:]>maxval*mean_spectrum)) & (ivar[fib,:]>0)
            nbad_it+=w.sum()
            nbad[fib]+=w.sum()

            if w.sum()>0:
                ivar[fib,w]=0
                log.warning("0th pass: masking {} pixels in fiber {}".format(w.sum(),fib))
            if nbad[fib]>=max_bad:
                ivar[fib,:]=0
                log.warning("0th pass: masking entire fiber {} (nbad={})".format(fib,nbad[fib]))
        if nbad_it == 0:
            break

    # 1st pass is median for spectrum, flat field without resolution
    # outlier rejection
    for iteration in range(max_iterations) :

        # use median for spectrum
        mean_spectrum=np.zeros((flux.shape[1]))
        for i in range(flux.shape[1]) :
            w=ivar[:,i]>0
            if w.sum() > 0 :
                mean_spectrum[i]=np.median(flux[w,i])

        nbad_it=0
        sum_chi2 = 0
        # not more than max_rej_it pixels per fiber at a time
        for fib in range(nfibers) :
            w=ivar[fib,:]>0
            if w.sum()==0:
                continue
            F = flux[fib,:]*0
            w=(mean_spectrum!=0) & (ivar[fib,:]>0)
            F[w]= flux[fib,w]/mean_spectrum[w]
            smooth_fiberflat[fib,:] = spline_fit(wave,wave[w],F[w],smoothing_res,ivar[fib,w]*mean_spectrum[w]**2)
            chi2 = ivar[fib,:]*(flux[fib,:]-mean_spectrum*smooth_fiberflat[fib,:])**2
            w=np.isnan(chi2)
            bad=np.where(chi2>nsig_clipping**2)[0]
            if bad.size>0 :
                if bad.size>max_rej_it : # not more than 5 pixels at a time
                    ii=np.argsort(chi2[bad])
                    bad=bad[ii[-max_rej_it:]]
                ivar[fib,bad] = 0
                log.warning("1st pass: rejecting {} pixels from fiber {}".format(len(bad),fib))
                nbad[fib]+=len(bad)
                if nbad[fib]>=max_bad:
                    ivar[fib,:]=0
                    log.warning("1st pass: rejecting fiber {} due to too many (new) bad pixels".format(fib))
                nbad_it+=len(bad)

            sum_chi2+=chi2.sum()
        ndf=int((ivar>0).sum()-nwave-nfibers*(nwave/smoothing_res))
        chi2pdf=0.
        if ndf>0 :
            chi2pdf=sum_chi2/ndf
        log.info("1st pass iter #{} chi2={}/{} chi2pdf={} nout={} (nsig={})".format(iteration,sum_chi2,ndf,chi2pdf,nbad_it,nsig_clipping))

        if nbad_it == 0 :
            break
    ## flatten fiberflat
    ## normalize smooth_fiberflat:
    mean=np.ones(smooth_fiberflat.shape[1])
    for i in range(smooth_fiberflat.shape[1]):
        w=ivar[:,i]>0
        if w.sum()>0:
            mean[i]=np.median(smooth_fiberflat[w,i])
    smooth_fiberflat = smooth_fiberflat/mean

    median_spectrum = mean_spectrum*1.

    previous_smooth_fiberflat = smooth_fiberflat*0
    log.info("after 1st pass : nout = %d/%d"%(np.sum(ivar==0),np.size(ivar.flatten())))
    # 2nd pass is full solution including deconvolved spectrum, no outlier rejection
    for iteration in range(max_iterations) :
        ## reset sum_chi2
        sum_chi2=0
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
        A_pos_def = A.todense()
        log.info("deconvolving")
        w = A.diagonal() > 0

        A_pos_def = A_pos_def[w,:]
        A_pos_def = A_pos_def[:,w]
        mean_spectrum = np.zeros(nwave)
        try:
            mean_spectrum[w]=cholesky_solve(A_pos_def,B[w])
        except:
            mean_spectrum[w]=np.linalg.lstsq(A_pos_def,B[w])[0]
            log.info("cholesky failes, trying svd inverse in iter {}".format(iteration))

        for fiber in range(nfibers) :

            if np.sum(ivar[fiber]>0)==0 :
                continue

            ### R = Resolution(resolution_data[fiber])
            R = frame.R[fiber]

            M = R.dot(mean_spectrum)
            ok=(M!=0) & (ivar[fiber,:]>0)
            if ok.sum()==0:
                continue
            smooth_fiberflat[fiber]=spline_fit(wave,wave[ok],flux[fiber,ok]/M[ok],smoothing_res,ivar[fiber,ok]*M[ok]**2)*(ivar[fiber,:]*M**2>0)
            chi2 = ivar[fiber]*(flux[fiber]-smooth_fiberflat[fiber]*M)**2
            sum_chi2 += chi2.sum()
            w=np.isnan(smooth_fiberflat[fiber])
            if w.sum()>0:
                ivar[fiber]=0
                smooth_fiberflat[fiber]=1

        # normalize to get a mean fiberflat=1
        mean = np.ones(smooth_fiberflat.shape[1])
        for i in range(nwave):
            w = ivar[:,i]>0
            if w.sum()>0:
                mean[i]=np.median(smooth_fiberflat[w,i])
        ok=np.where(mean!=0)[0]
        smooth_fiberflat[:,ok] /= mean[ok]

        # this is the max difference between two iterations
        max_diff=np.max(np.abs(smooth_fiberflat-previous_smooth_fiberflat)*(ivar>0.))
        previous_smooth_fiberflat=smooth_fiberflat.copy()

        ndf=int(np.sum(ivar>0)-nwave-nfibers*(nwave/smoothing_res))
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
            w=fiberflat_ivar[fiber,:]>0
            if w.sum()<100:
                break
            smooth_fiberflat=spline_fit(wave,wave[w],fiberflat[fiber,w],smoothing_res,fiberflat_ivar[fiber,w])
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

            ok=fiberflat_ivar[fiber]>0
            if ok.sum()==0:
                continue
            try:
                smooth_fiberflat=spline_fit(x,x[ok],fiberflat[fiber,ok],smoothing_res,fiberflat_ivar[fiber,ok])
                fiberflat[fiber,bad] = smooth_fiberflat[bad]
            except:
                fiberflat[fiber,bad] = 1
                fiberflat_ivar[fiber,bad]=0

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
        log.warning("High rejection rate: {:d}".format(qadict['N_MASK']))

    # Scale (search for low/high throughput)
    gdp = fiberflat.mask == 0
    rtio = frame.flux / np.outer(np.ones(fiberflat.nspec),fiberflat.meanspec)
    scale = np.median(rtio*gdp,axis=1)
    MAX_SCALE_OFF = float(np.max(np.abs(scale-1.)))
    fiber = int(np.argmax(np.abs(scale-1.)))
    qadict['MAX_SCALE_OFF'] = [MAX_SCALE_OFF, fiber]
    if qadict['MAX_SCALE_OFF'][0] > param['MAX_SCALE_OFF']:
        log.warning("Discrepant flux in fiberflat: {:g}, {:d}".format(
                qadict['MAX_SCALE_OFF'][0], qadict['MAX_SCALE_OFF'][1]))

    # Offset in fiberflat
    qadict['MAX_OFF'] = float(np.max(np.abs(fiberflat.fiberflat-1.)))
    if qadict['MAX_OFF'] > param['MAX_OFF']:
        log.warning("Large offset in fiberflat: {:g}".format(qadict['MAX_OFF']))

    # Offset in mean of fiberflat
    mean = np.mean(fiberflat.fiberflat*gdp,axis=1)
    fiber = int(np.argmax(np.abs(mean-1.)))
    qadict['MAX_MEAN_OFF'] = [float(np.max(np.abs(mean-1.))), fiber]
    if qadict['MAX_MEAN_OFF'][0] > param['MAX_MEAN_OFF']:
        log.warning("Discrepant mean in fiberflat: {:g}, {:d}".format(
                qadict['MAX_MEAN_OFF'][0], qadict['MAX_MEAN_OFF'][1]))

    # RMS in individual fibers
    rms = np.std(gdp*(fiberflat.fiberflat-
                      np.outer(mean, np.ones(fiberflat.nwave))),axis=1)
    fiber = int(np.argmax(rms))
    qadict['MAX_RMS'] = [float(np.max(rms)), fiber]
    if qadict['MAX_RMS'][0] > param['MAX_RMS']:
        log.warning("Large RMS in fiberflat: {:g}, {:d}".format(
                qadict['MAX_RMS'][0], qadict['MAX_RMS'][1]))

    # Return
    return qadict
