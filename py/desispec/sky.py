"""
desispec.sky
============

Utility functions to compute a sky model and subtract it.
"""


import numpy as np
from collections import OrderedDict

from desispec.resolution import Resolution
from desispec.linalg import cholesky_solve
from desispec.linalg import cholesky_invert
from desispec.linalg import spline_fit
from desiutil.log import get_logger
from desispec import util
from desiutil import stats as dustat
import scipy,scipy.sparse,scipy.stats,scipy.ndimage
from scipy.signal import fftconvolve
import sys
from desispec.fiberbitmasking import get_fiberbitmasked_frame_arrays, get_fiberbitmasked_frame
import scipy.ndimage
from desispec.maskbits import specmask

def compute_sky(frame, nsig_clipping=4.,max_iterations=100,model_ivar=False,add_variance=True,angular_variation_deg=0,chromatic_variation_deg=0,\
                adjust_wavelength=False,adjust_lsf=False,\
                only_use_skyfibers_for_adjustments=True,pcacorr=None) :
    """Compute a sky model.

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

    Optional:
        max_iterations : int , number of iterations
        model_ivar : replace ivar by a model to avoid bias due to correlated flux and ivar. this has a negligible effect on sims.
        add_variance : evaluate calibration error and add this to the sky model variance
        angular_variation_deg  : Degree of polynomial for sky flux variation with focal plane coordinates (default=0, i.e. no correction, a uniform sky)
        chromatic_variation_deg : Wavelength degree for the chromatic x angular terms. If negative, use as many 2D polynomials of x and y as wavelength entries.
        adjust_wavelength : adjust the wavelength of the sky model on sky lines to improve the sky subtraction
        adjust_lsf : adjust the LSF width of the sky model on sky lines to improve the sky subtraction
        only_use_skyfibers_for_adjustments: interpolate adjustments using sky fibers only
        pcacorr : SkyCorrPCA object to interpolate the wavelength or LSF adjustment from sky fibers to all fibers

    returns SkyModel object with attributes wave, flux, ivar, mask
    """
    if angular_variation_deg == 0 :
        skymodel = compute_uniform_sky(frame, nsig_clipping=nsig_clipping,max_iterations=max_iterations,\
                                       model_ivar=model_ivar,add_variance=add_variance,\
                                       adjust_wavelength=adjust_wavelength,adjust_lsf=adjust_lsf,\
                                       only_use_skyfibers_for_adjustments=only_use_skyfibers_for_adjustments,pcacorr=pcacorr)
    else :

        if adjust_wavelength :
            raise RuntimeError("combination of wavelength calibration adjustment and angular variations not yet implemented")
        if adjust_lsf :
            raise RuntimeError("combination of lsf calibration adjustment and angular variations not yet implemented")

        if chromatic_variation_deg < 0 :
            skymodel = compute_non_uniform_sky(frame, nsig_clipping=nsig_clipping,max_iterations=max_iterations,model_ivar=model_ivar,add_variance=add_variance,angular_variation_deg=angular_variation_deg)
        else :
            skymodel = compute_polynomial_times_sky(frame, nsig_clipping=nsig_clipping,max_iterations=max_iterations,model_ivar=model_ivar,add_variance=add_variance,angular_variation_deg=angular_variation_deg,chromatic_variation_deg=chromatic_variation_deg)

    skymodel.throughput_corrections = calculate_throughput_corrections(frame,skymodel)
    return skymodel

def _model_variance(frame,cskyflux,cskyivar,skyfibers) :
    """look at chi2 per wavelength and increase sky variance to reach chi2/ndf=1
    """

    log = get_logger()


    tivar = util.combine_ivar(frame.ivar[skyfibers], cskyivar[skyfibers])

    # the chi2 at a given wavelength can be large because on a cosmic
    # and not a psf error or sky non uniformity
    # so we need to consider only waves for which
    # a reasonable sky model error can be computed

    # mean sky
    msky = np.mean(cskyflux,axis=0)
    dwave = np.mean(np.gradient(frame.wave))
    dskydw = np.zeros(msky.shape)
    dskydw[1:-1]=(msky[2:]-msky[:-2])/(frame.wave[2:]-frame.wave[:-2])
    dskydw = np.abs(dskydw)

    # now we consider a worst possible sky model error (20% error on flat, 0.5A )
    max_possible_var = 1./(tivar+(tivar==0)) + (0.2*msky)**2 + (0.5*dskydw)**2

    # exclude residuals inconsistent with this max possible variance (at 3 sigma)
    bad = (frame.flux[skyfibers]-cskyflux[skyfibers])**2 > 3**2*max_possible_var
    tivar[bad]=0
    ndata = np.sum(tivar>0,axis=0)
    ok=np.where(ndata>1)[0]

    chi2  = np.zeros(frame.wave.size)
    chi2[ok] = np.sum(tivar*(frame.flux[skyfibers]-cskyflux[skyfibers])**2,axis=0)[ok]/(ndata[ok]-1)
    chi2[ndata<=1] = 1. # default

    # now we are going to evaluate a sky model error based on this chi2,
    # but only around sky flux peaks (>0.1*max)
    tmp   = np.zeros(frame.wave.size)
    tmp   = (msky[1:-1]>msky[2:])*(msky[1:-1]>msky[:-2])*(msky[1:-1]>0.1*np.max(msky))
    peaks = np.where(tmp)[0]+1
    dpix  = 2 #eval error range
    dpix2  = 3 # scale error range (larger)

    input_skyvar = 1./(cskyivar+(cskyivar==0))
    skyvar = input_skyvar + 0.

    # loop on peaks
    for peak in peaks :
        b=peak-dpix
        e=peak+dpix+1
        b2=peak-dpix2
        e2=peak+dpix2+1
        mchi2  = np.mean(chi2[b:e]) # mean reduced chi2 around peak
        mndata = np.mean(ndata[b:e]) # mean number of fibers contributing

        # sky model variance = sigma_flat * msky  + sigma_wave * dmskydw
        sigma_flat=0.005 # the fiber flat error is already included in the flux ivar, but empirical evidence we need an extra term
        sigma_wave=0.005 # A, minimum value
        res2=(frame.flux[skyfibers,b:e]-cskyflux[skyfibers,b:e])**2
        var=1./(tivar[:,b:e]+(tivar[:,b:e]==0))
        nd=np.sum(tivar[:,b:e]>0)
        sigma_wave = np.arange(0.005, 2, 0.005)

        #- pivar has shape (nskyfibers, npix, nsigma_wave)
        pivar = (tivar[:, b:e, np.newaxis]>0)/((var+(sigma_flat*msky[b:e])**2)[..., np.newaxis] + ((sigma_wave[np.newaxis,:]*dskydw[b:e, np.newaxis])**2)[np.newaxis, ...])
        #- chi2_of_sky_fibers has shape (nskyfibers, nsigma_wave)
        chi2_of_sky_fibers = np.sum(pivar*res2[..., np.newaxis],axis=1)/np.sum(tivar[:,b:e]>0,axis=1)[:, np.newaxis]
        #- normalization from median to mean for chi2 with 3 d.o.f.
        norm = 0.7888
        #- median_chi2 has shape (nsigma_wave,)
        median_chi2 = np.median(chi2_of_sky_fibers, axis=0)/norm
        if np.any(median_chi2 <= 1):
            #- first sigma_wave with median_chi2 <= 1 is the peak
            sigma_wave_peak = sigma_wave[np.where(median_chi2 <= 1)[0][0]]
        else :
            sigma_wave_peak = 2.
        log.info("peak at {}A : sigma_wave={}".format(int(frame.wave[peak]),sigma_wave_peak))
        skyvar[:,b2:e2] = input_skyvar[:,b2:e2] + (sigma_flat*msky[b2:e2])**2 + (sigma_wave_peak*dskydw[b2:e2])**2

    return (cskyivar>0)/(skyvar+(skyvar==0))



def compute_uniform_sky(frame, nsig_clipping=4.,max_iterations=100,model_ivar=False,add_variance=True,\
                        adjust_wavelength=True,adjust_lsf=True,only_use_skyfibers_for_adjustments = True, pcacorr=None) :

    """Compute a sky model.

    Sky[fiber,i] = R[fiber,i,j] Flux[j]

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

    Optional:
        max_iterations : int , number of iterations
        model_ivar : replace ivar by a model to avoid bias due to correlated flux and ivar. this has a negligible effect on sims.
        add_variance : evaluate calibration error and add this to the sky model variance
        adjust_wavelength : adjust the wavelength of the sky model on sky lines to improve the sky subtraction
        adjust_lsf : adjust the LSF width of the sky model on sky lines to improve the sky subtraction
        only_use_skyfibers_for_adjustments : interpolate adjustments using sky fibers only
        pcacorr : SkyCorrPCA object to interpolate the wavelength or LSF adjustment from sky fibers to all fibers

    returns SkyModel object with attributes wave, flux, ivar, mask
    """

    log=get_logger()
    log.info("starting")

    # Grab sky fibers on this frame
    skyfibers = np.where(frame.fibermap['OBJTYPE'] == 'SKY')[0]
    assert np.max(skyfibers) < 500  #- indices, not fiber numbers

    #- Hack: test tile 81097 (observed 20210430/00086750) had set
    #- FIBERSTATUS bit UNASSIGNED for sky targets on stuck positioners.
    #- Undo that.
    if (frame.meta is not None) and ('TILEID' in frame.meta) and (frame.meta['TILEID'] == 81097):
        log.info('Unsetting FIBERSTATUS UNASSIGNED for tileid 81097 sky fibers')
        frame.fibermap['FIBERSTATUS'][skyfibers] &= ~1

    nwave=frame.nwave

    current_ivar = get_fiberbitmasked_frame_arrays(frame,bitmask='sky',ivar_framemask=True,return_mask=False)

    # checking ivar because some sky fibers have been disabled
    bad=(np.sum(current_ivar[skyfibers]>0,axis=1)==0)
    good=~bad

    if np.any(bad) :
        log.warning("{} sky fibers discarded (because ivar=0 or bad FIBERSTATUS), only {} left.".format(np.sum(bad),np.sum(good)))
        skyfibers = skyfibers[good]

    if np.sum(good)==0 :
        message = "no valid sky fibers"
        log.error(message)
        raise RuntimeError(message)

    nfibers=len(skyfibers)

    current_ivar = current_ivar[skyfibers]
    flux = frame.flux[skyfibers]
    Rsky = frame.R[skyfibers]

    input_ivar=None
    if model_ivar :
        log.info("use a model of the inverse variance to remove bias due to correlated ivar and flux")
        input_ivar=current_ivar.copy()
        median_ivar_vs_wave  = np.median(current_ivar,axis=0)
        median_ivar_vs_fiber = np.median(current_ivar,axis=1)
        median_median_ivar   = np.median(median_ivar_vs_fiber)
        for f in range(current_ivar.shape[0]) :
            threshold=0.01
            current_ivar[f] = median_ivar_vs_fiber[f]/median_median_ivar * median_ivar_vs_wave
            # keep input ivar for very low weights
            ii=(input_ivar[f]<=(threshold*median_ivar_vs_wave))
            #log.info("fiber {} keep {}/{} original ivars".format(f,np.sum(ii),current_ivar.shape[1]))
            current_ivar[f][ii] = input_ivar[f][ii]


    sqrtw=np.sqrt(current_ivar)
    sqrtwflux=sqrtw*flux

    chi2=np.zeros(flux.shape)

    bad_skyfibers = []


    #max_iterations=2 ; log.warning("DEBUGGING LIMITING NUMBER OF ITERATIONS")

    nout_tot=0
    for iteration in range(max_iterations) :

        # the matrix A is 1/2 of the second derivative of the chi2 with respect to the parameters
        # A_ij = 1/2 d2(chi2)/di/dj
        # A_ij = sum_fiber sum_wave_w ivar[fiber,w] d(model)/di[fiber,w] * d(model)/dj[fiber,w]

        # the vector B is 1/2 of the first derivative of the chi2 with respect to the parameters
        # B_i  = 1/2 d(chi2)/di
        # B_i  = sum_fiber sum_wave_w ivar[fiber,w] d(model)/di[fiber,w] * (flux[fiber,w]-model[fiber,w])

        # the model is model[fiber]=R[fiber]*sky
        # and the parameters are the unconvolved sky flux at the wavelength i

        # so, d(model)/di[fiber,w] = R[fiber][w,i]
        # this gives
        # A_ij = sum_fiber  sum_wave_w ivar[fiber,w] R[fiber][w,i] R[fiber][w,j]
        # A = sum_fiber ( diag(sqrt(ivar))*R[fiber] ) ( diag(sqrt(ivar))* R[fiber] )^t
        # A = sum_fiber sqrtwR[fiber] sqrtwR[fiber]^t
        # and
        # B = sum_fiber sum_wave_w ivar[fiber,w] R[fiber][w] * flux[fiber,w]
        # B = sum_fiber sum_wave_w sqrt(ivar)[fiber,w]*flux[fiber,w] sqrtwR[fiber,wave]

        #A=scipy.sparse.lil_matrix((nwave,nwave)).tocsr()
        A=scipy.sparse.csr_matrix((nwave,nwave))
        B=np.zeros((nwave))

        # diagonal sparse matrix with content = sqrt(ivar)*flat of a given fiber
        SD=scipy.sparse.dia_matrix((nwave,nwave))

        # loop on fiber to handle resolution
        for fiber in range(nfibers) :
            if fiber%10==0 :
                log.info("iter %d sky fiber %d/%d"%(iteration,fiber,nfibers))
            R = Rsky[fiber]

            # diagonal sparse matrix with content = sqrt(ivar)
            SD.setdiag(sqrtw[fiber])

            sqrtwR = SD*R # each row r of R is multiplied by sqrtw[r]
            A += sqrtwR.T*sqrtwR
            B += sqrtwR.T*sqrtwflux[fiber]
        A = A.toarray()

        log.info("iter %d solving"%iteration)
        w = A.diagonal()>0
        A_pos_def = A[w,:]
        A_pos_def = A_pos_def[:,w]
        deconvolved_sky = B*0
        try:
            deconvolved_sky[w]=cholesky_solve(A_pos_def,B[w])
        except:
            log.info("cholesky failed, trying svd in iteration {}".format(iteration))
            deconvolved_sky[w]=np.linalg.lstsq(A_pos_def,B[w])[0]

        log.info("iter %d compute chi2"%iteration)

        medflux=np.zeros(nfibers)
        for fiber in range(nfibers) :
            # the parameters are directly the unconvolve sky flux
            # so we simply have to reconvolve it
            fiber_convolved_sky_flux = Rsky[fiber].dot(deconvolved_sky)
            chi2[fiber]=current_ivar[fiber]*(flux[fiber]-fiber_convolved_sky_flux)**2
            ok=(current_ivar[fiber]>0)
            if np.sum(ok)>0 :
                medflux[fiber] = np.median((flux[fiber]-fiber_convolved_sky_flux)[ok])

        log.info("rejecting")

        # whole fiber with excess flux
        if np.sum(medflux!=0) > 2 : # at least 3 valid sky fibers
            rms_from_nmad = 1.48*np.median(np.abs(medflux[medflux!=0]))
            # discard fibers that are 7 sigma away
            badfibers=np.where(np.abs(medflux)>7*rms_from_nmad)[0]
            for fiber in badfibers :
                log.warning("discarding fiber {} with median flux = {:.2f} > 7*{:.2f}".format(skyfibers[fiber],medflux[fiber],rms_from_nmad))
                current_ivar[fiber]=0
                sqrtw[fiber]=0
                sqrtwflux[fiber]=0
                # set a mask bit here
                bad_skyfibers.append(skyfibers[fiber])
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


    # we know have to compute the sky model for all fibers
    # and propagate the uncertainties

    # no need to restore the original ivar to compute the model errors when modeling ivar
    # the sky inverse variances are very similar

    log.info("compute the parameter covariance")
    # we may have to use a different method to compute this
    # covariance

    try :
        parameter_covar=cholesky_invert(A)
        # the above is too slow
        # maybe invert per block, sandwich by R
    except np.linalg.linalg.LinAlgError :
        log.warning("cholesky_solve_and_invert failed, switching to np.linalg.lstsq and np.linalg.pinv")
        parameter_covar = np.linalg.pinv(A)

    log.info("compute mean resolution")
    # we make an approximation for the variance to save CPU time
    # we use the average resolution of all fibers in the frame:
    mean_res_data=np.mean(frame.resolution_data,axis=0)
    Rmean = Resolution(mean_res_data)

    log.info("compute convolved sky and ivar")

    # The parameters are directly the unconvolved sky
    # First convolve with average resolution :
    convolved_sky_covar=Rmean.dot(parameter_covar).dot(Rmean.T.todense())

    # and keep only the diagonal
    convolved_sky_var=np.diagonal(convolved_sky_covar)

    # inverse
    convolved_sky_ivar=(convolved_sky_var>0)/(convolved_sky_var+(convolved_sky_var==0))

    # and simply consider it's the same for all spectra
    cskyivar = np.tile(convolved_sky_ivar, frame.nspec).reshape(frame.nspec, nwave)

    # The sky model for each fiber (simple convolution with resolution of each fiber)
    cskyflux = np.zeros(frame.flux.shape)
    for i in range(frame.nspec):
        cskyflux[i] = frame.R[i].dot(deconvolved_sky)

    # See if we can improve the sky model by readjusting the wavelength and/or the width of sky lines
    if adjust_wavelength or adjust_lsf :
        log.info("adjust the wavelength of sky spectrum on sky lines to improve sky subtraction ...")

        if adjust_wavelength :
            # compute derivative of sky w.r.t. wavelength
            dskydwave = np.gradient(cskyflux,axis=1)/np.gradient(frame.wave)
        else :
            dskydwave = None

        if adjust_lsf :
            # compute derivative of sky w.r.t. lsf width
            dwave = np.mean(np.gradient(frame.wave))
            dsigma_A   = 0.3 #A
            dsigma_bin = dsigma_A/dwave # consider this extra width for the PSF (sigma' = sqrt(sigma**2+dsigma**2))
            hw=int(4*dsigma_bin)+1
            x=np.arange(-hw,hw+1)
            k=np.zeros((3,x.size)) # a Gaussian kernel
            k[1]=np.exp(-x**2/dsigma_bin**2/2.)
            k/=np.sum(k)
            tmp = fftconvolve(cskyflux,k,mode="same")
            dskydlsf = (tmp-cskyflux)/dsigma_A # variation of line shape with width
        else :
            dskydlsf = None

        # detect peaks in mean sky spectrum
        # peaks = local maximum larger than 10% of max peak
        meansky = np.mean(cskyflux,axis=0)
        tmp   = (meansky[1:-1]>meansky[2:])*(meansky[1:-1]>meansky[:-2])*(meansky[1:-1]>0.1*np.max(meansky))
        peaks = np.where(tmp)[0]+1
        # remove edges
        peaks = peaks[(peaks>10)&(peaks<meansky.size-10)]
        peak_wave=frame.wave[peaks]

        log.info("Number of peaks: {}".format(peaks.size))
        if  peaks.size < 10 :
            log.info("Wavelength of peaks: {}".format(peak_wave))

        # define area around each sky line to adjust
        dwave = np.mean(np.gradient(frame.wave))
        dpix = int(3//dwave)+1

        # number of parameters to fit for each peak: delta_wave , delta_lsf , scale of sky , a background (to absorb source signal)
        nparam = 2
        if adjust_wavelength : nparam += 1
        if adjust_lsf : nparam += 1

        AA=np.zeros((nparam,nparam))
        BB=np.zeros((nparam))

        # temporary arrays with best fit parameters on peaks
        # for each fiber, with errors and chi2/ndf
        peak_scale=np.zeros((frame.nspec,peaks.size))
        peak_scale_err=np.zeros((frame.nspec,peaks.size))
        peak_dw=np.zeros((frame.nspec,peaks.size))
        peak_dw_err=np.zeros((frame.nspec,peaks.size))
        peak_dlsf=np.zeros((frame.nspec,peaks.size))
        peak_dlsf_err=np.zeros((frame.nspec,peaks.size))

        peak_chi2pdf=np.zeros((frame.nspec,peaks.size))

        # interpolated values across peaks, after selection
        # based on precision and chi2
        interpolated_sky_dwave=np.zeros(frame.flux.shape)
        interpolated_sky_dlsf=np.zeros(frame.flux.shape)

        # loop on fibers and then on sky spectrum peaks
        if only_use_skyfibers_for_adjustments :
            fibers_in_fit = skyfibers
        else :
            fibers_in_fit = np.arange(frame.nspec)

        # restrict to fibers with ivar!=0
        ok = np.sum(frame.ivar[fibers_in_fit],axis=1)>0
        fibers_in_fit = fibers_in_fit[ok]

        # loop on sky spectrum peaks, compute for all fibers simultaneously
        for j,peak in enumerate(peaks) :
            b = peak-dpix
            e = peak+dpix+1
            npix = e - b
            flux = frame.flux[fibers_in_fit][:,b:e]
            ivar = frame.ivar[fibers_in_fit][:,b:e]
            if b < 0 or e > frame.flux.shape[1] :
                log.warning("skip peak on edge of spectrum with b={} e={}".format(b,e))
                continue
            M = np.zeros((fibers_in_fit.size, nparam, npix))
            index = 0
            M[:, index] = np.ones(npix); index += 1
            M[:, index] = cskyflux[fibers_in_fit][:, b:e]; index += 1
            if adjust_wavelength : M[:, index] = dskydwave[fibers_in_fit][:, b:e]; index += 1
            if adjust_lsf        : M[:, index] = dskydlsf[fibers_in_fit][:, b:e]; index += 1
            # Solve (M * W * M.T) X = (M * W * flux)
            BB = np.einsum('ijk,ik->ij', M, ivar*flux)
            AA = np.einsum('ijk,ik,ilk->ijl', M, ivar, M)
            # solve linear system
            #- TODO: replace with X = np.linalg.solve(AA, BB) ?
            try:
                AAi=np.linalg.inv(AA)
            except np.linalg.LinAlgError as e:
                log.warning(str(e))
                continue
            # save best fit parameter and errors
            X = np.einsum('ijk,ik->ij', AAi, BB)
            X_err = np.sqrt(AAi*(AAi>0))
            index = 1
            peak_scale[fibers_in_fit,j] = X[:, index]
            peak_scale_err[fibers_in_fit,j] = X_err[:, index, index]
            index += 1
            if adjust_wavelength:
                peak_dw[fibers_in_fit, j] = X[:, index]
                peak_dw_err[fibers_in_fit, j] = X_err[:, index, index]
                index += 1
            if adjust_lsf:
                peak_dlsf[fibers_in_fit, j] = X[:, index]
                peak_dlsf_err[fibers_in_fit, j] = X_err[:, index, index]
                index += 1

            residuals = flux
            for index in range(nparam) :
                #for index in range(3) : # needed for compatibility with master (but this was a bug)
                residuals -= X[:,index][:, np.newaxis]*M[:,index]

            variance = 1.0/(ivar+(ivar==0)) + (0.05*M[:,1])**2
            peak_chi2pdf[fibers_in_fit, j] = np.sum((ivar>0)/variance*(residuals)**2, axis=1)/(npix-nparam)

        for i in fibers_in_fit :
            # for each fiber, select valid peaks and interpolate
            ok=(peak_chi2pdf[i]<2)
            if adjust_wavelength :
                ok &= (peak_dw_err[i]>0.)&(peak_dw_err[i]<0.1) # error on wavelength shift
            if adjust_lsf :
                ok &= (peak_dlsf_err[i]>0.)&(peak_dlsf_err[i]<0.3) # error on line width (quadratic, so 0.3 mean a change of width of 0.3**2/2~5%)
            # piece-wise linear interpolate across the whole spectrum between the sky line peaks
            # this interpolation will be used to alter the whole sky spectrum
            if np.sum(ok)>0 :
                if adjust_wavelength :
                    interpolated_sky_dwave[i]=np.interp(frame.wave,peak_wave[ok],peak_dw[i,ok])
                if adjust_lsf :
                    interpolated_sky_dlsf[i]=np.interp(frame.wave,peak_wave[ok],peak_dlsf[i,ok])
                line=""
                if adjust_wavelength :
                    line += " dlambda mean={:4.3f} rms={:4.3f} A".format(np.mean(interpolated_sky_dwave[i]),np.std(interpolated_sky_dwave[i]))
                if adjust_lsf :
                    line += " dlsf mean={:4.3f} rms={:4.3f} A".format(np.mean(interpolated_sky_dlsf[i]),np.std(interpolated_sky_dlsf[i]))
                log.info(line)

        # we ignore the interpolated_sky_scale which is too sensitive
        # to CCD defects or cosmic rays

        if pcacorr is None :
            if only_use_skyfibers_for_adjustments :
                goodfibers=fibers_in_fit
            else : # keep all except bright objects and interpolate over them
                mflux=np.median(frame.flux,axis=1)
                mmflux=np.median(mflux)
                rms=1.48*np.median(np.abs(mflux-mmflux))
                selection=(mflux<mmflux+2*rms)
                # at least 80% of good pixels
                ngood=np.sum((frame.ivar>0)*(frame.mask==0),axis=1)
                selection &= (ngood>0.8*frame.flux.shape[1])
                goodfibers=np.where(mflux<mmflux+2*rms)[0]
                log.info("number of good fibers=",goodfibers.size)
            allfibers=np.arange(frame.nspec)
            # the actual median filtering
            if adjust_wavelength :
                for j in range(interpolated_sky_dwave.shape[1]) :
                    interpolated_sky_dwave[:,j] = np.interp(np.arange(interpolated_sky_dwave.shape[0]),goodfibers,interpolated_sky_dwave[goodfibers,j])
                cskyflux += interpolated_sky_dwave*dskydwave
            if adjust_lsf : # simple interpolation over fibers
                for j in range(interpolated_sky_dlsf.shape[1]) :
                    interpolated_sky_dlsf[:,j] = np.interp(np.arange(interpolated_sky_dlsf.shape[0]),goodfibers,interpolated_sky_dlsf[goodfibers,j])
                cskyflux += interpolated_sky_dlsf*dskydlsf

        else :


            def fit_and_interpolate(delta,skyfibers,mean,components,label="") :
                mean_and_components = np.zeros((components.shape[0]+1,
                                                components.shape[1],
                                                components.shape[2]))
                mean_and_components[0]  = mean
                mean_and_components[1:] = components
                ncomp=mean_and_components.shape[0]
                log.info("Will fit a linear combination on {} components for {}".format(ncomp,label))
                AA=np.zeros((ncomp,ncomp))
                BB=np.zeros(ncomp)
                for i in range(ncomp) :
                    BB[i] = np.sum(delta[skyfibers]*mean_and_components[i][skyfibers])
                    for j in range(i,ncomp) :
                        AA[i,j] = np.sum(mean_and_components[i][skyfibers]*mean_and_components[j][skyfibers])
                        if j!=i :
                            AA[j,i]=AA[i,j]
                AAi=np.linalg.inv(AA)
                X=AAi.dot(BB)
                log.info("Best fit linear coefficients for {} = {}".format(label,list(X)))
                result = np.zeros_like(delta)
                for i in range(ncomp) :
                    result += X[i]*mean_and_components[i]
                return result


            # we are going to fit a linear combination of the PCA coefficients only on the sky fibers
            # and then apply the linear combination to all fibers
            log.info("Use PCA skycorr")

            if adjust_wavelength :
                correction = fit_and_interpolate(interpolated_sky_dwave,skyfibers,\
                                                 pcacorr.dwave_mean,pcacorr.dwave_eigenvectors,label="wavelength")
                cskyflux  += correction*dskydwave
            if adjust_lsf :
                correction = fit_and_interpolate(interpolated_sky_dlsf,skyfibers,\
                                                 pcacorr.dlsf_mean,pcacorr.dlsf_eigenvectors,label="LSF")
                cskyflux  += correction*dskydlsf

    # look at chi2 per wavelength and increase sky variance to reach chi2/ndf=1
    if skyfibers.size > 1 and add_variance :
        modified_cskyivar = _model_variance(frame,cskyflux,cskyivar,skyfibers)
    else :
        modified_cskyivar = cskyivar.copy()

    # set sky flux and ivar to zero to poorly constrained regions
    # and add margins to avoid expolation issues with the resolution matrix
    wmask = (np.diagonal(A)<=0).astype(float)
    # empirically, need to account for the full width of the resolution band
    # (realized here by applying twice the resolution)
    wmask = Rmean.dot(Rmean.dot(wmask))
    bad = np.where(wmask!=0)[0]
    cskyflux[:,bad]=0.
    modified_cskyivar[:,bad]=0.

    # minimum number of fibers at each wavelength
    min_number_of_fibers = min(10,max(1,skyfibers.size//2))
    fibers_with_signal=np.sum(current_ivar>0,axis=0)
    bad = (fibers_with_signal<min_number_of_fibers)
    # increase by 1 pixel
    bad[1:-1] |= bad[2:]
    bad[1:-1] |= bad[:-2]
    cskyflux[:,bad]=0.
    modified_cskyivar[:,bad]=0.


    mask = (modified_cskyivar==0).astype(np.uint32)

    # add mask bits for bad sky fibers
    bad_skyfibers = np.unique(bad_skyfibers)
    if bad_skyfibers.size > 0 :
        mask[bad_skyfibers] |= specmask.mask("BADSKY")

    skymodel = SkyModel(frame.wave.copy(), cskyflux, modified_cskyivar, mask,
                        nrej=nout_tot, stat_ivar = cskyivar) # keep a record of the statistical ivar for QA
    if adjust_wavelength :
        skymodel.dwave = interpolated_sky_dwave
    if adjust_lsf :
        skymodel.dlsf  = interpolated_sky_dlsf

    return skymodel


def compute_polynomial_times_sky(frame, nsig_clipping=4.,max_iterations=30,model_ivar=False,add_variance=True,angular_variation_deg=1,chromatic_variation_deg=1) :
    """Compute a sky model.

    Sky[fiber,i] = R[fiber,i,j] Polynomial(x[fiber],y[fiber],wavelength[j]) Flux[j]

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

    Optional:
        max_iterations : int , number of iterations
        model_ivar : replace ivar by a model to avoid bias due to correlated flux and ivar. this has a negligible effect on sims.
        add_variance : evaluate calibration error and add this to the sky model variance

    returns SkyModel object with attributes wave, flux, ivar, mask
    """

    log=get_logger()
    log.info("starting")

    # Grab sky fibers on this frame
    skyfibers = np.where(frame.fibermap['OBJTYPE'] == 'SKY')[0]
    assert np.max(skyfibers) < 500  #- indices, not fiber numbers

    nwave=frame.nwave
    nfibers=len(skyfibers)

    current_ivar = get_fiberbitmasked_frame_arrays(frame,bitmask='sky',ivar_framemask=True,return_mask=False)
    current_ivar = current_ivar[skyfibers]
    flux = frame.flux[skyfibers]

    Rsky = frame.R[skyfibers]

    input_ivar=None
    if model_ivar :
        log.info("use a model of the inverse variance to remove bias due to correlated ivar and flux")
        input_ivar=current_ivar.copy()
        median_ivar_vs_wave  = np.median(current_ivar,axis=0)
        median_ivar_vs_fiber = np.median(current_ivar,axis=1)
        median_median_ivar   = np.median(median_ivar_vs_fiber)
        for f in range(current_ivar.shape[0]) :
            threshold=0.01
            current_ivar[f] = median_ivar_vs_fiber[f]/median_median_ivar * median_ivar_vs_wave
            # keep input ivar for very low weights
            ii=(input_ivar[f]<=(threshold*median_ivar_vs_wave))
            #log.info("fiber {} keep {}/{} original ivars".format(f,np.sum(ii),current_ivar.shape[1]))
            current_ivar[f][ii] = input_ivar[f][ii]

    # need focal plane coordinates
    x = frame.fibermap["FIBERASSIGN_X"]
    y = frame.fibermap["FIBERASSIGN_Y"]

    # normalize for numerical stability
    xm = np.mean(x)
    ym = np.mean(y)
    xs = np.std(x)
    ys = np.std(y)
    if xs==0 : xs = 1
    if ys==0 : ys = 1
    x = (x-xm)/xs
    y = (y-ym)/ys
    w = (frame.wave-frame.wave[0])/(frame.wave[-1]-frame.wave[0])*2.-1

    # precompute the monomials for the sky fibers
    log.debug("compute monomials for deg={} and {}".format(angular_variation_deg,chromatic_variation_deg))
    monomials=[]
    for dx in range(angular_variation_deg+1) :
        for dy in range(angular_variation_deg+1-dx) :
            xypol = (x**dx)*(y**dy)
            for dw in range(chromatic_variation_deg+1) :
                wpol=w**dw
                monomials.append(np.outer(xypol,wpol))

    ncoef=len(monomials)
    coef=np.zeros((ncoef))

    allfibers_monomials=np.array(monomials)
    log.debug("shape of allfibers_monomials = {}".format(allfibers_monomials.shape))

    skyfibers_monomials = allfibers_monomials[:,skyfibers,:]
    log.debug("shape of skyfibers_monomials = {}".format(skyfibers_monomials.shape))


    sqrtw=np.sqrt(current_ivar)
    sqrtwflux=sqrtw*flux

    chi2=np.zeros(flux.shape)

    Pol     = np.ones(flux.shape,dtype=float)
    coef[0] = 1.

    nout_tot=0
    previous_chi2=-10.
    for iteration in range(max_iterations) :

        # the matrix A is 1/2 of the second derivative of the chi2 with respect to the parameters
        # A_ij = 1/2 d2(chi2)/di/dj
        # A_ij = sum_fiber sum_wave_w ivar[fiber,w] d(model)/di[fiber,w] * d(model)/dj[fiber,w]

        # the vector B is 1/2 of the first derivative of the chi2 with respect to the parameters
        # B_i  = 1/2 d(chi2)/di
        # B_i  = sum_fiber sum_wave_w ivar[fiber,w] d(model)/di[fiber,w] * (flux[fiber,w]-model[fiber,w])

        # the model is model[fiber]=R[fiber]*Pol(x,y,wave)*sky
        # the parameters are the unconvolved sky flux at the wavelength i
        # and the polynomial coefficients

        A=scipy.sparse.csr_matrix((nwave,nwave),dtype=float)
        B=np.zeros((nwave),dtype=float)
        D=scipy.sparse.dia_matrix((nwave,nwave))
        D2=scipy.sparse.dia_matrix((nwave,nwave))

        Pol /= coef[0] # force constant term to 1.

        # solving for the deconvolved mean sky spectrum
        # loop on fiber to handle resolution
        for fiber in range(nfibers) :
            if fiber%10==0 :
                log.info("iter %d sky fiber (1st fit) %d/%d"%(iteration,fiber,nfibers))
            D.setdiag(sqrtw[fiber])
            D2.setdiag(Pol[fiber])
            sqrtwRP = D.dot(Rsky[fiber]).dot(D2) # each row r of R is multiplied by sqrtw[r]
            A += sqrtwRP.T*sqrtwRP
            B += sqrtwRP.T*sqrtwflux[fiber]
        A = A.toarray()

        log.info("iter %d solving"%iteration)
        w = A.diagonal()>0
        A_pos_def = A[w,:]
        A_pos_def = A_pos_def[:,w]
        parameters = B*0
        try:
            parameters[w]=cholesky_solve(A_pos_def,B[w])
        except:
            log.info("cholesky failed, trying svd in iteration {}".format(iteration))
            parameters[w]=np.linalg.lstsq(A_pos_def,B[w])[0]
        # parameters = the deconvolved mean sky spectrum

        # now evaluate the polynomial coefficients
        Ap=np.zeros((ncoef,ncoef),dtype=float)
        Bp=np.zeros((ncoef),dtype=float)
        D2.setdiag(parameters)
        for fiber in range(nfibers) :
            if fiber%10==0 :
                log.info("iter %d sky fiber  (2nd fit) %d/%d"%(iteration,fiber,nfibers))
            D.setdiag(sqrtw[fiber])
            sqrtwRSM = D.dot(Rsky[fiber]).dot(D2).dot(skyfibers_monomials[:,fiber,:].T)
            Ap += sqrtwRSM.T.dot(sqrtwRSM)
            Bp += sqrtwRSM.T.dot(sqrtwflux[fiber])

        # Add huge prior on zeroth angular order terms to converge faster
        # (because those terms are degenerate with the mean deconvolved spectrum)
        weight=1e24
        Ap[0,0] += weight
        Bp[0]   += weight # force 0th term to 1
        for i in range(1,chromatic_variation_deg+1) :
            Ap[i,i] += weight # force other wavelength terms to 0


        coef=cholesky_solve(Ap,Bp)
        log.info("pol coef = {}".format(coef))

        # recompute the polynomial values
        Pol = skyfibers_monomials.T.dot(coef).T

        # chi2 and outlier rejection
        log.info("iter %d compute chi2"%iteration)
        for fiber in range(nfibers) :
            chi2[fiber]=current_ivar[fiber]*(flux[fiber]-Rsky[fiber].dot(Pol[fiber]*parameters))**2

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

        log.info("iter #%d chi2=%g ndf=%d chi2pdf=%f delta=%f nout=%d"%(iteration,sum_chi2,ndf,chi2pdf,abs(sum_chi2-previous_chi2),nout_iter))

        if nout_iter == 0 and abs(sum_chi2-previous_chi2)<0.2 :
            break
        previous_chi2 = sum_chi2+0.


    log.info("nout tot=%d"%nout_tot)

    # we know have to compute the sky model for all fibers
    # and propagate the uncertainties

    # no need to restore the original ivar to compute the model errors when modeling ivar
    # the sky inverse variances are very similar

    # we ignore here the fact that we have fit a angular variation,
    # so the sky model uncertainties are inaccurate

    log.info("compute the parameter covariance")
    try :
        parameter_covar=cholesky_invert(A)
    except np.linalg.linalg.LinAlgError :
        log.warning("cholesky_solve_and_invert failed, switching to np.linalg.lstsq and np.linalg.pinv")
        parameter_covar = np.linalg.pinv(A)

    log.info("compute mean resolution")
    # we make an approximation for the variance to save CPU time
    # we use the average resolution of all fibers in the frame:
    mean_res_data=np.mean(frame.resolution_data,axis=0)
    Rmean = Resolution(mean_res_data)

    log.info("compute convolved sky and ivar")

    # The parameters are directly the unconvolved sky
    # First convolve with average resolution :
    convolved_sky_covar=Rmean.dot(parameter_covar).dot(Rmean.T.todense())

    # and keep only the diagonal
    convolved_sky_var=np.diagonal(convolved_sky_covar)

    # inverse
    convolved_sky_ivar=(convolved_sky_var>0)/(convolved_sky_var+(convolved_sky_var==0))

    # and simply consider it's the same for all spectra
    cskyivar = np.tile(convolved_sky_ivar, frame.nspec).reshape(frame.nspec, nwave)

    # The sky model for each fiber (simple convolution with resolution of each fiber)
    cskyflux = np.zeros(frame.flux.shape)

    Pol = allfibers_monomials.T.dot(coef).T
    for fiber in range(frame.nspec):
        cskyflux[fiber] = frame.R[fiber].dot(Pol[fiber]*parameters)

    # look at chi2 per wavelength and increase sky variance to reach chi2/ndf=1
    if skyfibers.size > 1 and add_variance :
        modified_cskyivar = _model_variance(frame,cskyflux,cskyivar,skyfibers)
    else :
        modified_cskyivar = cskyivar.copy()

    # set sky flux and ivar to zero to poorly constrained regions
    # and add margins to avoid expolation issues with the resolution matrix
    wmask = (np.diagonal(A)<=0).astype(float)
    # empirically, need to account for the full width of the resolution band
    # (realized here by applying twice the resolution)
    wmask = Rmean.dot(Rmean.dot(wmask))
    bad = np.where(wmask!=0)[0]
    cskyflux[:,bad]=0.
    modified_cskyivar[:,bad]=0.

    # minimum number of fibers at each wavelength
    min_number_of_fibers = min(10,max(1,skyfibers.size//2))
    fibers_with_signal=np.sum(current_ivar>0,axis=0)
    bad = (fibers_with_signal<min_number_of_fibers)
    # increase by 1 pixel
    bad[1:-1] |= bad[2:]
    bad[1:-1] |= bad[:-2]
    cskyflux[:,bad]=0.
    modified_cskyivar[:,bad]=0.


    # need to do better here
    mask = (modified_cskyivar==0).astype(np.uint32)

    return SkyModel(frame.wave.copy(), cskyflux, modified_cskyivar, mask,
                    nrej=nout_tot, stat_ivar = cskyivar) # keep a record of the statistical ivar for QA


def compute_non_uniform_sky(frame, nsig_clipping=4.,max_iterations=10,model_ivar=False,add_variance=True,angular_variation_deg=1) :
    """Compute a sky model.

    Sky[fiber,i] = R[fiber,i,j] ( Flux_0[j] + x[fiber]*Flux_x[j] + y[fiber]*Flux_y[j] + ... )

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

    Optional:
        max_iterations : int , number of iterations
        model_ivar : replace ivar by a model to avoid bias due to correlated flux and ivar. this has a negligible effect on sims.
        add_variance : evaluate calibration error and add this to the sky model variance
        angular_variation_deg  : degree of 2D polynomial correction as a function of fiber focal plane coordinates (default=1). One set of coefficients per wavelength

    returns SkyModel object with attributes wave, flux, ivar, mask
    """

    log=get_logger()
    log.info("starting")

    # Grab sky fibers on this frame
    skyfibers = np.where(frame.fibermap['OBJTYPE'] == 'SKY')[0]
    assert np.max(skyfibers) < 500  #- indices, not fiber numbers

    nwave=frame.nwave
    nfibers=len(skyfibers)

    current_ivar = get_fiberbitmasked_frame_arrays(frame,bitmask='sky',ivar_framemask=True,return_mask=False)
    current_ivar *= (frame.mask==0)

    current_ivar = current_ivar[skyfibers]
    flux = frame.flux[skyfibers]
    Rsky = frame.R[skyfibers]


    # need focal plane coordinates of fibers
    x = frame.fibermap["FIBERASSIGN_X"][skyfibers]
    y = frame.fibermap["FIBERASSIGN_Y"][skyfibers]
    # normalize for numerical stability
    xm = np.mean(frame.fibermap["FIBERASSIGN_X"])
    ym = np.mean(frame.fibermap["FIBERASSIGN_Y"])
    xs = np.std(frame.fibermap["FIBERASSIGN_X"])
    ys = np.std(frame.fibermap["FIBERASSIGN_Y"])
    if xs==0 : xs = 1
    if ys==0 : ys = 1
    x = (x-xm)/xs
    y = (y-ym)/ys

    # precompute the monomials for the sky fibers
    log.debug("compute monomials for deg={}".format(angular_variation_deg))
    monomials=[]
    for dx in range(angular_variation_deg+1) :
        for dy in range(angular_variation_deg+1-dx) :
            monomials.append((x**dx)*(y**dy))
    ncoef=len(monomials)
    monomials=np.array(monomials)


    input_ivar=None
    if model_ivar :
        log.info("use a model of the inverse variance to remove bias due to correlated ivar and flux")
        input_ivar=current_ivar.copy()
        median_ivar_vs_wave  = np.median(current_ivar,axis=0)
        median_ivar_vs_fiber = np.median(current_ivar,axis=1)
        median_median_ivar   = np.median(median_ivar_vs_fiber)
        for f in range(current_ivar.shape[0]) :
            threshold=0.01
            current_ivar[f] = median_ivar_vs_fiber[f]/median_median_ivar * median_ivar_vs_wave
            # keep input ivar for very low weights
            ii=(input_ivar[f]<=(threshold*median_ivar_vs_wave))
            #log.info("fiber {} keep {}/{} original ivars".format(f,np.sum(ii),current_ivar.shape[1]))
            current_ivar[f][ii] = input_ivar[f][ii]


    sqrtw=np.sqrt(current_ivar)
    sqrtwflux=sqrtw*flux

    chi2=np.zeros(flux.shape)




    nout_tot=0
    for iteration in range(max_iterations) :

        # the matrix A is 1/2 of the second derivative of the chi2 with respect to the parameters
        # A_ij = 1/2 d2(chi2)/di/dj
        # A_ij = sum_fiber sum_wave_w ivar[fiber,w] d(model)/di[fiber,w] * d(model)/dj[fiber,w]

        # the vector B is 1/2 of the first derivative of the chi2 with respect to the parameters
        # B_i  = 1/2 d(chi2)/di
        # B_i  = sum_fiber sum_wave_w ivar[fiber,w] d(model)/di[fiber,w] * (flux[fiber,w]-model[fiber,w])

        # with x_fiber,y_fiber the fiber coordinates in the focal plane (or sky)
        # the unconvolved sky flux at wavelength i is a polynomial of x_fiber,y_fiber
        # sky(fiber,i) = pol(x_fiber,y_fiber,p) = sum_p a_ip * x_fiber**degx(p) y_fiber**degy(p)
        # sky(fiber,i) =  sum_p monom[fiber,p] *  a_ip
        # the convolved sky flux at wavelength w is
        # model[fiber,w] = sum_i R[fiber][w,i] sum_p monom[fiber,p] *  a_ip
        # model[fiber,w] = sum_p monom[fiber,p] R[fiber][w,i] a_ip
        #
        # so, the matrix A is composed of blocks (p,k) corresponding to polynomial coefficient indices where
        # A[pk] = sum_fiber monom[fiber,p]*monom[fiber,k] sqrtwR[fiber] sqrtwR[fiber]^t
        # similarily
        # B[p]  =  sum_fiber monom[fiber,p] * sum_wave_w (sqrt(ivar)[fiber,w]*flux[fiber,w]) sqrtwR[fiber,wave]

        A=np.zeros((nwave*ncoef,nwave*ncoef))
        B=np.zeros((nwave*ncoef))

        # diagonal sparse matrix with content = sqrt(ivar)*flat of a given fiber
        SD=scipy.sparse.lil_matrix((nwave,nwave))

        # loop on fiber to handle resolution
        for fiber in range(nfibers) :
            if fiber%10==0 :
                log.info("iter %d sky fiber %d/%d"%(iteration,fiber,nfibers))
            R = Rsky[fiber]

            # diagonal sparse matrix with content = sqrt(ivar)
            SD.setdiag(sqrtw[fiber])

            sqrtwR = SD*R # each row r of R is multiplied by sqrtw[r]

            #wRtR=(sqrtwR.T*sqrtwR).tocsr()
            wRtR=(sqrtwR.T*sqrtwR).todense()
            wRtF=sqrtwR.T*sqrtwflux[fiber]
            # loop on polynomial coefficients (double loop for A)
            # fill only blocks of A and B
            for p in range(ncoef) :
                for k in range(ncoef) :
                    A[p*nwave:(p+1)*nwave,k*nwave:(k+1)*nwave] += monomials[p,fiber]*monomials[k,fiber]*wRtR
                B[p*nwave:(p+1)*nwave] += monomials[p,fiber]*wRtF

        log.info("iter %d solving"%iteration)
        w = A.diagonal()>0
        A_pos_def = A[w,:]
        A_pos_def = A_pos_def[:,w]
        parameters = B*0
        try:
            parameters[w]=cholesky_solve(A_pos_def,B[w])
        except:
            log.info("cholesky failed, trying svd in iteration {}".format(iteration))
            parameters[w]=np.linalg.lstsq(A_pos_def,B[w])[0]

        log.info("iter %d compute chi2"%iteration)

        for fiber in range(nfibers) :
            # loop on polynomial indices
            unconvolved_fiber_sky_flux = np.zeros(nwave)
            for p in range(ncoef) :
                unconvolved_fiber_sky_flux += monomials[p,fiber]*parameters[p*nwave:(p+1)*nwave]
            # then convolve
            fiber_convolved_sky_flux = Rsky[fiber].dot(unconvolved_fiber_sky_flux)

            chi2[fiber]=current_ivar[fiber]*(flux[fiber]-fiber_convolved_sky_flux)**2

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


    # we know have to compute the sky model for all fibers
    # and propagate the uncertainties

    # no need to restore the original ivar to compute the model errors when modeling ivar
    # the sky inverse variances are very similar

    # is there a different method to compute this ?
    log.info("compute covariance")
    try :
        parameter_covar=cholesky_invert(A)
    except np.linalg.linalg.LinAlgError :
        log.warning("cholesky_solve_and_invert failed, switching to np.linalg.lstsq and np.linalg.pinv")
        parameter_covar = np.linalg.pinv(A)

    log.info("compute mean resolution")
    # we make an approximation for the variance to save CPU time
    # we use the average resolution of all fibers in the frame:
    mean_res_data=np.mean(frame.resolution_data,axis=0)
    Rmean = Resolution(mean_res_data)

    log.info("compute convolved sky and ivar")

    cskyflux = np.zeros(frame.flux.shape)
    cskyivar = np.zeros(frame.flux.shape)

    log.info("compute convolved parameter covariance")
    # The covariance of the parameters is composed of ncoef*ncoef blocks each of size nwave*nwave
    # A block (p,k) is the covariance of the unconvolved spectra p and k , corresponding to the polynomial indices p and k
    # We first sandwich each block with the average resolution.
    convolved_parameter_covar=np.zeros((ncoef,ncoef,nwave))
    for p in range(ncoef) :
        for k in range(ncoef) :
            convolved_parameter_covar[p,k] = np.diagonal(Rmean.dot(parameter_covar[p*nwave:(p+1)*nwave,k*nwave:(k+1)*nwave]).dot(Rmean.T.todense()))

    '''
    import astropy.io.fits as pyfits
    pyfits.writeto("convolved_parameter_covar.fits",convolved_parameter_covar,overwrite=True)

    # other approach
    log.info("dense Rmean...")
    Rmean=Rmean.todense()
    log.info("invert Rinv...")
    Rinv=np.linalg.inv(Rmean)
    # check this
    print("0?",np.max(np.abs(Rinv.dot(Rmean)-np.eye(Rmean.shape[0])))/np.max(np.abs(Rmean)))
    convolved_parameter_ivar=np.zeros((ncoef,ncoef,nwave))
    for p in range(ncoef) :
        for k in range(ncoef) :
            convolved_parameter_ivar[p,k] = np.diagonal(Rinv.T.dot(A[p*nwave:(p+1)*nwave,k*nwave:(k+1)*nwave]).dot(Rinv))
    # solve for each wave separately
    convolved_parameter_covar=np.zeros((ncoef,ncoef,nwave))
    for i in range(nwave) :
        print("inverting ivar of wave %d/%d"%(i,nwave))
        convolved_parameter_covar[:,:,i] = cholesky_invert(convolved_parameter_ivar[:,:,i])
    pyfits.writeto("convolved_parameter_covar_bis.fits",convolved_parameter_covar,overwrite=True)
    import sys
    sys.exit(12)
    '''

    # Now we compute the sky model variance for each fiber individually
    # accounting for its focal plane coordinates
    # so that a target fiber distant for a sky fiber will naturally have a larger
    # sky model variance
    log.info("compute sky and variance per fiber")
    for i in range(frame.nspec):
        # compute monomials
        M = []
        xi=(frame.fibermap["FIBERASSIGN_X"][i]-xm)/xs
        yi=(frame.fibermap["FIBERASSIGN_Y"][i]-ym)/ys
        for dx in range(angular_variation_deg+1) :
            for dy in range(angular_variation_deg+1-dx) :
                M.append((xi**dx)*(yi**dy))
        M = np.array(M)

        unconvolved_fiber_sky_flux=np.zeros(nwave)
        convolved_fiber_skyvar=np.zeros(nwave)
        for p in range(ncoef) :
            unconvolved_fiber_sky_flux += M[p]*parameters[p*nwave:(p+1)*nwave]
            for k in range(ncoef) :
                convolved_fiber_skyvar += M[p]*M[k]*convolved_parameter_covar[p,k]

        # convolve sky model with this fiber's resolution
        cskyflux[i] = frame.R[i].dot(unconvolved_fiber_sky_flux)

        # save inverse of variance
        cskyivar[i] = (convolved_fiber_skyvar>0)/(convolved_fiber_skyvar+(convolved_fiber_skyvar==0))


    # look at chi2 per wavelength and increase sky variance to reach chi2/ndf=1
    if skyfibers.size > 1 and add_variance :
        modified_cskyivar = _model_variance(frame,cskyflux,cskyivar,skyfibers)
    else :
        modified_cskyivar = cskyivar.copy()

    # set sky flux and ivar to zero to poorly constrained regions
    # and add margins to avoid expolation issues with the resolution matrix
    wmask = (np.diagonal(A[:nwave,:nwave])<=0).astype(float)
    # empirically, need to account for the full width of the resolution band
    # (realized here by applying twice the resolution)
    wmask = Rmean.dot(Rmean.dot(wmask))
    bad = np.where(wmask!=0)[0]
    cskyflux[:,bad]=0.
    modified_cskyivar[:,bad]=0.

    # minimum number of fibers at each wavelength
    min_number_of_fibers = min(10,max(1,skyfibers.size//2))
    fibers_with_signal=np.sum(current_ivar>0,axis=0)
    bad = (fibers_with_signal<min_number_of_fibers)
    # increase by 1 pixel
    bad[1:-1] |= bad[2:]
    bad[1:-1] |= bad[:-2]
    cskyflux[:,bad]=0.
    modified_cskyivar[:,bad]=0.


    # need to do better here
    mask = (modified_cskyivar==0).astype(np.uint32)

    return SkyModel(frame.wave.copy(), cskyflux, modified_cskyivar, mask,
                    nrej=nout_tot, stat_ivar = cskyivar) # keep a record of the statistical ivar for QA

class SkyModel(object):
    def __init__(self, wave, flux, ivar, mask, header=None, nrej=0, stat_ivar=None, throughput_corrections=None):
        """Create SkyModel object

        Args:
            wave  : 1D[nwave] wavelength in Angstroms
            flux  : 2D[nspec, nwave] sky model to subtract
            ivar  : 2D[nspec, nwave] inverse variance of the sky model
            mask  : 2D[nspec, nwave] 0=ok or >0 if problems; 32-bit
            header : (optional) header from FITS file HDU0
            nrej : (optional) Number of rejected pixels in fit
            throughput_corrections : 1D (optional) Multiplicative throughput corrections for each fiber
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
        self.stat_ivar = stat_ivar
        self.throughput_corrections = throughput_corrections
        self.dwave = None # wavelength corrections
        self.dlsf  = None # LSF corrections

def subtract_sky(frame, skymodel, apply_throughput_correction = True, zero_ivar=True) :
    """Subtract skymodel from frame, altering frame.flux, .ivar, and .mask

    Args:
        frame : desispec.Frame object
        skymodel : desispec.SkyModel object

    Option:
        apply_throughput_correction : if True, fit for an achromatic throughput correction.
                                      This is to absorb variations of Focal Ratio Degradation with fiber flexure.

        zero_ivar : if True , set ivar=0 for masked pixels
    """
    assert frame.nspec == skymodel.nspec
    assert frame.nwave == skymodel.nwave

    log=get_logger()
    log.info("starting with apply_throughput_correction = {} and zero_ivar = {}".format(apply_throughput_correction, zero_ivar))

    # Set fibermask flagged spectra to have 0 flux and variance
    frame = get_fiberbitmasked_frame(frame,bitmask='sky',ivar_framemask=zero_ivar)

    # check same wavelength, die if not the case
    if not np.allclose(frame.wave, skymodel.wave):
        message = "frame and sky not on same wavelength grid"
        log.error(message)
        raise ValueError(message)

    if apply_throughput_correction and skymodel.throughput_corrections is not None :
        # need to fit for a multiplicative factor of the sky model
        # before subtraction
        # we are going to use a set of bright sky lines,
        # and fit a multiplicative factor + background around
        # each of them individually, and then combine the results
        # with outlier rejection in case a source emission line
        # coincides with one of the sky lines.
        for fiber in range(frame.flux.shape[0]) :
            # apply this correction to the sky model even if we have not fit it (default can be 1 or 0)
            skymodel.flux[fiber] *= skymodel.throughput_corrections[fiber]

    frame.flux -= skymodel.flux
    frame.ivar = util.combine_ivar(frame.ivar, skymodel.ivar)
    frame.mask |= skymodel.mask

    log.info("done")

def calculate_throughput_corrections(frame,skymodel):
    """
    Calculate the throughput corrections for each fiber based on the skymodel.

    Args:
        frame (Frame object): frame containing the data that may need to be corrected
        skymodel (SkyModel object): skymodel object that contains the information about the sky for the given exposure/frame

    Output:
        corrections (1D array):  1D array where the index corresponds to the fiber % 500 and the values are the multiplicative corrections that would
                             be applied to the fluxes in frame.flux to correct them based on the input skymodel
    """
    # need to fit for a multiplicative factor of the sky model
    # before subtraction
    # we are going to use a set of bright sky lines,
    # and fit a multiplicative factor + background around
    # each of them individually, and then combine the results
    # with outlier rejection in case a source emission line
    # coincides with one of the sky lines.

    # it's more robust to have a hardcoded set of sky lines here
    # these are all the sky lines with a flux >5% of the max flux
    # except in b where we add an extra weaker line at 5199.4A
    skyline=np.array([5199.4,5578.4,5656.4,5891.4,5897.4,6302.4,6308.4,6365.4,6500.4,6546.4,\
                      6555.4,6618.4,6663.4,6679.4,6690.4,6765.4,6831.4,6836.4,6865.4,6925.4,\
                      6951.4,6980.4,7242.4,7247.4,7278.4,7286.4,7305.4,7318.4,7331.4,7343.4,\
                      7360.4,7371.4,7394.4,7404.4,7440.4,7526.4,7714.4,7719.4,7752.4,7762.4,\
                      7782.4,7796.4,7810.4,7823.4,7843.4,7855.4,7862.4,7873.4,7881.4,7892.4,\
                      7915.4,7923.4,7933.4,7951.4,7966.4,7982.4,7995.4,8016.4,8028.4,8064.4,\
                      8280.4,8284.4,8290.4,8298.4,8301.4,8313.4,8346.4,8355.4,8367.4,8384.4,\
                      8401.4,8417.4,8432.4,8454.4,8467.4,8495.4,8507.4,8627.4,8630.4,8634.4,\
                      8638.4,8652.4,8657.4,8662.4,8667.4,8672.4,8677.4,8683.4,8763.4,8770.4,\
                      8780.4,8793.4,8829.4,8835.4,8838.4,8852.4,8870.4,8888.4,8905.4,8922.4,\
                      8945.4,8960.4,8990.4,9003.4,9040.4,9052.4,9105.4,9227.4,9309.4,9315.4,\
                      9320.4,9326.4,9340.4,9378.4,9389.4,9404.4,9422.4,9442.4,9461.4,9479.4,\
                      9505.4,9521.4,9555.4,9570.4,9610.4,9623.4,9671.4,9684.4,9693.4,9702.4,\
                      9714.4,9722.4,9740.4,9748.4,9793.4,9802.4,9814.4,9820.4])

    # half width of wavelength region around each sky line
    # larger values give a better statistical precision
    # but also a larger sensitivity to source features
    # best solution on one dark night exposure obtained with
    # a half width of 4A.
    hw=4#A
    tivar=frame.ivar
    if frame.mask is not None :
        tivar *= (frame.mask==0)
        tivar *= (skymodel.ivar>0)

    # we precompute the quantities needed to fit each sky line + continuum
    # the sky "line profile" is the actual sky model
    # and we consider an additive constant
    sw,swf,sws,sws2,swsf=[],[],[],[],[]
    for line in skyline :
        if line<=frame.wave[0] or line>=frame.wave[-1] : continue
        ii=np.where((frame.wave>=line-hw)&(frame.wave<=line+hw))[0]
        if ii.size<2 : continue
        sw.append(np.sum(tivar[:,ii],axis=1))
        swf.append(np.sum(tivar[:,ii]*frame.flux[:,ii],axis=1))
        swsf.append(np.sum(tivar[:,ii]*frame.flux[:,ii]*skymodel.flux[:,ii],axis=1))
        sws.append(np.sum(tivar[:,ii]*skymodel.flux[:,ii],axis=1))
        sws2.append(np.sum(tivar[:,ii]*skymodel.flux[:,ii]**2,axis=1))

    log=get_logger()
    nlines=len(sw)
    corrections = np.ones(frame.flux.shape[0]).astype('f8')
    for fiber in range(frame.flux.shape[0]) :
        # we solve the 2x2 linear system for each fiber and sky line
        # and save the results for each fiber
        coef=[] # list of scale values
        var=[] # list of variance on scale values
        for line in range(nlines) :
            if sw[line][fiber]<=0 : continue
            A=np.array([[sw[line][fiber],sws[line][fiber]],[sws[line][fiber],sws2[line][fiber]]])
            B=np.array([swf[line][fiber],swsf[line][fiber]])
            try :
                Ai=np.linalg.inv(A)
                X=Ai.dot(B)
                coef.append(X[1]) # the scale coef (marginalized over cst background)
                var.append(Ai[1,1])
            except :
                pass

        if len(coef)==0 :
            log.warning("cannot corr. throughput. for fiber %d"%fiber)
            continue

        coef=np.array(coef)
        var=np.array(var)
        ivar=(var>0)/(var+(var==0)+0.005**2)
        ivar_for_outliers=(var>0)/(var+(var==0)+0.02**2)

        # loop for outlier rejection
        failed=False
        for loop in range(50) :
            a=np.sum(ivar)
            if a <= 0 :
                log.warning("cannot corr. throughput. ivar=0 everywhere on sky lines for fiber %d"%fiber)
                failed=True
                break

            mcoef=np.sum(ivar*coef)/a
            mcoeferr=1/np.sqrt(a)

            nsig=3.
            chi2=ivar_for_outliers*(coef-mcoef)**2
            worst=np.argmax(chi2)
            if chi2[worst]>nsig**2*np.median(chi2[chi2>0]) : # with rough scaling of errors
                #log.debug("discard a bad measurement for fiber %d"%(fiber))
                ivar[worst]=0
                ivar_for_outliers[worst]=0
            else :
                break

        if failed :
            continue


        log.info("fiber #%03d throughput corr = %5.4f +- %5.4f (mean fiber flux=%f)"%(fiber,mcoef,mcoeferr,np.median(frame.flux[fiber])))
        if mcoeferr>0.1 :
            log.warning("throughput corr error = %5.4f > 0.1 is too large for fiber %d, do not apply correction"%(mcoeferr,fiber))
        else :
            corrections[fiber] = mcoef

    return corrections


def qa_skysub(param, frame, skymodel, quick_look=False):
    """Calculate QA on SkySubtraction

    Note: Pixels rejected in generating the SkyModel (as above), are
    not rejected in the stats calculated here.  Would need to carry
    along current_ivar to do so.

    Args:
        param : dict of QA parameters : see qa_frame.init_skysub for example
        frame : desispec.Frame object;  Should have been flat fielded
        skymodel : desispec.SkyModel object
        quick_look : bool, optional
          If True, do QuickLook specific QA (or avoid some)
    Returns:
        qadict: dict of QA outputs
          Need to record simple Python objects for yaml (str, float, int)
    """
    from desispec.qa import qalib
    import copy

    log=get_logger()

    #- QAs
    #- first subtract sky to get the sky subtracted frame. This is only for QA. Pipeline does it separately.
    tempframe=copy.deepcopy(frame) #- make a copy so as to propagate frame unaffected so that downstream pipeline uses it.
    subtract_sky(tempframe,skymodel) #- Note: sky subtract is done to get residuals. As part of pipeline it is done in fluxcalib stage

    # Sky residuals first
    qadict = qalib.sky_resid(param, tempframe, skymodel, quick_look=quick_look)

    # Sky continuum
    if not quick_look:  # Sky continuum is measured after flat fielding in QuickLook
        channel = frame.meta['CAMERA'][0]
        wrange1, wrange2 = param[channel.upper()+'_CONT']
        skyfiber, contfiberlow, contfiberhigh, meancontfiber, skycont = qalib.sky_continuum(frame,wrange1,wrange2)
        qadict["SKYFIBERID"] = skyfiber.tolist()
        qadict["SKYCONT"] = skycont
        qadict["SKYCONT_FIBER"] = meancontfiber

    if quick_look:  # The following can be a *large* dict
        qadict_snr = qalib.SignalVsNoise(tempframe,param)
        qadict.update(qadict_snr)

    return qadict
