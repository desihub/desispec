"""
desispec.fluxcalibration
========================

Flux calibration routines.
"""
from __future__ import absolute_import
import numpy as np
from .resolution import Resolution
from .linalg import cholesky_solve, cholesky_solve_and_invert, spline_fit
from .interpolation import resample_flux
from .log import get_logger
from .io.filters import load_filter
from desispec import util
import scipy, scipy.sparse, scipy.ndimage
import sys
import time
from astropy import units
import multiprocessing

#rebin spectra into new wavebins. This should be equivalent to desispec.interpolation.resample_flux. So may not be needed here
#But should move from here anyway.

def rebinSpectra(spectra,oldWaveBins,newWaveBins):
    tck=scipy.interpolate.splrep(oldWaveBins,spectra,s=0,k=1)
    specnew=scipy.interpolate.splev(newWaveBins,tck,der=0)
    return specnew

def applySmoothingFilter(flux):
    return scipy.ndimage.filters.median_filter(flux,200)
#
# Import some global constants.
#
# Why not use astropy constants?
#
# This is VERY inconvenient when trying to build documentation!
# The documentation may be build in an environment that does not have
# scipy installed.  There is no obvious reason why this has to be a module-level
# calculation.
#
import scipy.constants as const
h=const.h
pi=const.pi
e=const.e
c=const.c
erg=const.erg
try:
    hc = const.h/const.erg*const.c*1.e10  # (in units of ergsA)
except TypeError:
    hc = 1.9864458241717586e-08

def compute_chi2(wave,normalized_flux,normalized_ivar,resolution_data,shifted_stdwave,star_stdflux) :
    chi2 = None
    try :
        chi2=0.
        for cam in normalized_flux.keys() :
            tmp=resample_flux(wave[cam],shifted_stdwave,star_stdflux) # this is slow
            model=Resolution(resolution_data[cam]).dot(tmp) # this is slow
            tmp=applySmoothingFilter(model) # this is fast
            normalized_model = model/(tmp+(tmp==0))
            chi2 += np.sum(normalized_ivar[cam]*(normalized_flux[cam]-normalized_model)**2)
    except :
        chi2 = 1e20
    return chi2

def _func(arg) :
    return compute_chi2(**arg)

def match_templates(wave, flux, ivar, resolution_data, stdwave, stdflux, teff, logg, feh, ncpu=1, z_max=0.005, z_res=0.00005):
    """For each input spectrum, identify which standard star template is the closest
    match, factoring out broadband throughput/calibration differences.

    Args:
        wave : A dictionary of 1D array of vacuum wavelengths [Angstroms]. Example below.
        flux : A dictionary of 1D observed flux for the star
        ivar : A dictionary 1D inverse variance of flux
        resolution_data: resolution corresponding to the star's fiber
        stdwave : 1D standard star template wavelengths [Angstroms]
        stdflux : 2D[nstd, nwave] template flux
        teff : 1D[nstd] effective model temperature
        logg : 1D[nstd] model surface gravity
        feh : 1D[nstd] model metallicity
        ncpu : number of cpu for multiprocessing

    Returns:
        index : index of standard star
        redshift : redshift of standard star
        chipdf : reduced chi2

    Notes:
      - wave and stdwave can be on different grids that don't
        necessarily overlap
      - wave does not have to be uniform or monotonic.  Multiple cameras
        can be supported by concatenating their wave and flux arrays
    """
    # I am treating the input arguments from three frame files as dictionary. For example
    # wave{"r":rwave,"b":bwave,"z":zwave}
    # Each data(3 channels) is compared to every model.

    # flux should be already flat fielded and sky subtracted.
    # First normalize both data and model by dividing by median filter.

    cameras = flux.keys()
    log = get_logger()
    log.debug(time.asctime())

    # find canonical f-type model: Teff=6000, logg=4, Fe/H=-1.5
    #####################################
    canonical_model=np.argmin((teff-6000.0)**2+(logg-4.0)**2+(feh+1.5)**2)
    #log.info("canonical model=%s"%str(canonical_model))

    # resampling on a log wavelength grid
    #####################################
    # need to go fast at the beginning ... so we resample both data and model on a log grid

    # define grid
    minwave = 100000.
    maxwave = 0.
    for cam in cameras :
        minwave=min(minwave,np.min(wave[cam]))
        maxwave=max(maxwave,np.max(wave[cam]))
    # ala boss
    lstep=np.log10(1+z_res)
    margin=int(np.log10(1+z_max)/lstep)+1
    minlwave=np.log10(minwave)
    maxlwave=np.log10(maxwave) # desired, but readjusted
    nstep=(maxlwave-minlwave)/lstep
    #print "nstep=",nstep
    resampled_lwave=minlwave+lstep*np.arange(nstep)
    resampled_wave=10**resampled_lwave

    # map data on grid
    resampled_data={}
    resampled_ivar={}
    resampled_model={}
    for cam in cameras :
        tmp_flux,tmp_ivar=resample_flux(resampled_wave,wave[cam],flux[cam],ivar[cam])
        resampled_data[cam]=tmp_flux
        resampled_ivar[cam]=tmp_ivar

        # we need to have the model on a larger grid than the data wave for redshifting
        dwave=wave[cam][-1]-wave[cam][-2]
        npix=int((wave[cam][-1]*z_max)/dwave+2)
        extended_cam_wave=np.append( wave[cam][0]+dwave*np.arange(-npix,0) ,  wave[cam])
        extended_cam_wave=np.append( extended_cam_wave, wave[cam][-1]+dwave*np.arange(1,npix+1))
        # ok now we also need to increase the resolution
        tmp_res=np.zeros((resolution_data[cam].shape[0],resolution_data[cam].shape[1]+2*npix))
        tmp_res[:,:npix] = np.tile(resolution_data[cam][:,0],(npix,1)).T
        tmp_res[:,npix:-npix] = resolution_data[cam]
        tmp_res[:,-npix:] = np.tile(resolution_data[cam][:,-1],(npix,1)).T
        # resampled model at camera resolution, with margin
        tmp=resample_flux(extended_cam_wave,stdwave,stdflux[canonical_model])
        tmp=Resolution(tmp_res).dot(tmp)
        # map on log lam grid
        resampled_model[cam]=resample_flux(resampled_wave,extended_cam_wave,tmp)

        # we now normalize both model and data
        tmp=applySmoothingFilter(resampled_data[cam])
        resampled_data[cam]/=(tmp+(tmp==0))
        resampled_ivar[cam]*=tmp**2
        tmp=applySmoothingFilter(resampled_model[cam])
        resampled_model[cam]/=(tmp+(tmp==0))
        resampled_ivar[cam]*=(tmp!=0)

    # fit the best redshift
    chi2=np.zeros((2*margin+1))
    for i in range(-margin,margin+1) :
        for cam in cameras :
            if i<margin :
                chi2[i+margin] += np.sum(resampled_ivar[cam][margin:-margin]*(resampled_data[cam][margin:-margin]-resampled_model[cam][margin+i:-margin+i])**2)
            else :
                chi2[i+margin] += np.sum(resampled_ivar[cam][margin:-margin]*(resampled_data[cam][margin:-margin]-resampled_model[cam][margin+i:])**2)
    i=np.argmin(chi2)-margin
    z=10**(i*lstep)-1
    #log.info("Best z=%f"%z)

    normalized_flux={}
    normalized_ivar={}
    ndata=0
    for cam in cameras :
        tmp=applySmoothingFilter(flux[cam]) # this is fast
        normalized_flux[cam] = flux[cam]/(tmp+(tmp==0))
        normalized_ivar[cam] = ivar[cam]*tmp**2
        # mask potential cosmics
        ok=np.where(normalized_ivar[cam]>0)[0]
        if ok.size>0 :
            normalized_ivar[cam][ok] *= (normalized_flux[cam][ok]<1.+3/np.sqrt(normalized_ivar[cam][ok]))
        ndata += np.sum(normalized_ivar[cam]>0)




    # now we go back to the model spectra , redshift them, resample, apply resolution, normalize and chi2 match

    nstars=stdflux.shape[0]
    shifted_stdwave=stdwave/(1+z)

    func_args = []
    # need to parallelize this
    for star in range(nstars) :
        arguments={"wave":wave,
                   "normalized_flux":normalized_flux,
                   "normalized_ivar":normalized_ivar,
                   "resolution_data":resolution_data,
                   "shifted_stdwave":shifted_stdwave,
                   "star_stdflux":stdflux[star]}
        func_args.append( arguments )

    if ncpu > 1:
        log.debug("creating multiprocessing pool with %d cpus"%ncpu); sys.stdout.flush()
        pool = multiprocessing.Pool(ncpu)
        log.debug("Running pool.map() for {} items".format(len(func_args))); sys.stdout.flush()
        model_chi2 =  pool.map(_func, func_args)
        log.debug("Finished pool.map()"); sys.stdout.flush()
        pool.close()
        pool.join()
        log.debug("Finished pool.join()"); sys.stdout.flush()
    else:
        log.debug("Not using multiprocessing for {} cpus".format(ncpu))
        model_chi2 = [_func(x) for x in func_args]
        log.debug("Finished serial loop over compute_chi2")
        
    best_model_id=np.argmin(np.array(model_chi2))
    best_chi2=model_chi2[best_model_id]
    log.debug("selected best model {} chi2/ndf {}".format(best_model_id, best_chi2/ndata))
    # log.info("model star#%d chi2/ndf=%f best chi2/ndf=%f"%(star,chi2/ndata,best_chi2/ndata))

    return best_model_id,z,best_chi2/ndata


def normalize_templates(stdwave, stdflux, mags, filters):
    """Returns spectra normalized to input magnitudes.

    Args:
        stdwave : 1D array of standard star wavelengths [Angstroms]
        stdflux : 1D observed flux
        mags : 1D array of observed AB magnitudes
        filters : list of filter names for mags, e.g. ['SDSS_r', 'DECAM_g', ...]

    Returns:
        stdwave : same as input
        normflux : normalized flux array

    Only SDSS_r band is assumed to be used for normalization for now.
    """
    log = get_logger()

    nstdwave=stdwave.size
    normflux=np.array(nstdwave)

    fluxunits = 1e-17 * units.erg / units.s / units.cm**2 / units.Angstrom

    for i,v in enumerate(filters):
        #Normalizing using only SDSS_R band magnitude
        if v.upper() == 'SDSS_R' or v.upper() =='DECAM_R' or v.upper()=='DECAM_G' :
            #-TODO: Add more filters for calibration. Which one should be used if multiple mag available?
            refmag=mags[i]
            filter_response=load_filter(v)
            apMag=filter_response.get_ab_magnitude(stdflux*fluxunits,stdwave)
            log.info('scaling {} mag {:f} to {:f}.'.format(v, apMag,refmag))
            scalefac=10**((apMag-refmag)/2.5)
            normflux=stdflux*scalefac

            break  #- found SDSS_R or DECAM_R; we can stop now
        count=0
        for k,f in enumerate(['SDSS_R','DECAM_R','DECAM_G']):
            ii,=np.where((np.asarray(filters)==f))
            count=count+ii.shape[0]
        if (count==0):
            log.error("No magnitude given for SDSS_R, DECAM_R or DECAM_G filters")
            sys.exit(0)
    return normflux

def compute_flux_calibration(frame, input_model_wave,input_model_flux,nsig_clipping=4.,debug=False):
    """Compute average frame throughput based on data frame.(wave,flux,ivar,resolution_data)
    and spectro-photometrically calibrated stellar models (model_wave,model_flux).
    Wave and model_wave are not necessarily on the same grid

    Args:
      frame : Frame object with attributes wave, flux, ivar, resolution_data
      input_model_wave : 1D[nwave] array of model wavelengths
      input_model_flux : 2D[nstd, nwave] array of model fluxes
      nsig_clipping : (optional) sigma clipping level

    Returns:
         desispec.FluxCalib object
         calibration: mean calibration (without resolution)

    Notes:
      - we first resample the model on the input flux wave grid
      - then convolve it to the data resolution (the input wave grid is supposed finer than the spectral resolution)
      - then iteratively
        - fit the mean throughput (deconvolved, this is needed because of sharp atmospheric absorption lines)
        - compute broad band correction to fibers (to correct for small mis-alignement for instance)
        - perform outlier rejection

     There is one subtelty with the relation between calibration and resolution.
      - The input frame flux is on average flux^frame_fiber = R_fiber*C*flux^true where C is the true calibration (or throughput)
        which is a function of wavelength. This is the system we solve.
      - But we want to return a calibration vector per fiber C_fiber defined by flux^cframe_fiber = flux^frame_fiber/C_fiber,
        such that flux^cframe can be compared with a convolved model of the truth, flux^cframe_fiber = R_fiber*flux^true,
        i.e. (R_fiber*C*flux^true)/C_fiber = R_fiber*true_flux, giving C_fiber = (R_fiber*C*flux^true)/(R_fiber*flux^true)
      - There is no solution for this for all possible input specta. The solution for a flat spectrum is returned,
        which is very close to C_fiber = R_fiber*C (but not exactly).

    """

    log=get_logger()
    log.info("starting")

    #- Pull out just the standard stars for convenience, but keep the
    #- full frame of spectra around because we will later need to convolved
    #- the calibration vector for each fiber individually
    stdfibers = (frame.fibermap['OBJTYPE'] == 'STD')
    stdstars = frame[stdfibers]

    nwave=stdstars.nwave
    nstds=stdstars.flux.shape[0]

    # resample model to data grid and convolve by resolution
    model_flux=np.zeros((nstds, nwave))
    convolved_model_flux=np.zeros((nstds, nwave))
    for fiber in range(model_flux.shape[0]) :
        model_flux[fiber]=resample_flux(stdstars.wave,input_model_wave,input_model_flux[fiber])
        convolved_model_flux[fiber]=stdstars.R[fiber].dot(model_flux[fiber])

    # iterative fitting and clipping to get precise mean spectrum
    current_ivar=stdstars.ivar.copy()

    #- Start with a first pass median rejection
    median_calib = np.median(stdstars.flux / convolved_model_flux, axis=0)
    chi2 = stdstars.ivar * (stdstars.flux - convolved_model_flux*median_calib)**2
    bad=(chi2>nsig_clipping**2)
    current_ivar[bad] = 0

    smooth_fiber_correction=np.ones((stdstars.flux.shape))
    chi2=np.zeros((stdstars.flux.shape))

    # chi2 = sum w ( data_flux - R*(calib*model_flux))**2
    # chi2 = sum (sqrtw*data_flux -diag(sqrtw)*R*diag(model_flux)*calib)

    sqrtw=np.sqrt(current_ivar)
    #sqrtwmodel=np.sqrt(current_ivar)*convolved_model_flux # used only for QA
    sqrtwflux=np.sqrt(current_ivar)*stdstars.flux

    # diagonal sparse matrices
    D1=scipy.sparse.lil_matrix((nwave,nwave))
    D2=scipy.sparse.lil_matrix((nwave,nwave))

    # test
    # nstds=20
    nout_tot=0
    for iteration in range(20) :

        # fit mean calibration
        A=scipy.sparse.lil_matrix((nwave,nwave)).tocsr()
        B=np.zeros((nwave))

        # loop on fiber to handle resolution
        for fiber in range(nstds) :
            if fiber%10==0 :
                log.info("iter %d fiber %d"%(iteration,fiber))

            R = stdstars.R[fiber]

            # diagonal sparse matrix with content = sqrt(ivar)*flat
            D1.setdiag(sqrtw[fiber]*smooth_fiber_correction[fiber])
            D2.setdiag(model_flux[fiber])
            sqrtwmodelR = D1.dot(R.dot(D2)) # chi2 = sum (sqrtw*data_flux -diag(sqrtw)*smooth_fiber_correction*R*diag(model_flux)*calib )

            A = A+(sqrtwmodelR.T*sqrtwmodelR).tocsr()
            B += sqrtwmodelR.T*sqrtwflux[fiber]

        #- Add a weak prior that calibration = median_calib
        #- to keep A well conditioned
        minivar = np.min(current_ivar[current_ivar>0])
        log.debug('min(ivar[ivar>0]) = {}'.format(minivar))
        epsilon = minivar/10000
        A = epsilon*np.eye(nwave) + A   #- converts sparse A -> dense A
        B += median_calib*epsilon

        log.info("iter %d solving"%iteration)
        ### log.debug('cond(A) {:g}'.format(np.linalg.cond(A)))
        calibration=cholesky_solve(A, B)

        log.info("iter %d fit smooth correction per fiber"%iteration)
        # fit smooth fiberflat and compute chi2
        for fiber in range(nstds) :
            if fiber%10==0 :
                log.info("iter %d fiber %d(smooth)"%(iteration,fiber))

            M = stdstars.R[fiber].dot(calibration*model_flux[fiber])

            pol=np.poly1d(np.polyfit(stdstars.wave,stdstars.flux[fiber]/(M+(M==0)),deg=1,w=current_ivar[fiber]*M**2))
            smooth_fiber_correction[fiber]=pol(stdstars.wave)
            chi2[fiber]=current_ivar[fiber]*(stdstars.flux[fiber]-smooth_fiber_correction[fiber]*M)**2

        log.info("iter {0:d} rejecting".format(iteration))

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
                #sqrtwmodel[worst_entry,i]=0
                sqrtwflux[worst_entry,i]=0
                nout_iter += 1

        else :
            # remove all of them at once
            bad=(chi2>nsig_clipping**2)
            current_ivar *= (bad==0)
            sqrtw *= (bad==0)
            #sqrtwmodel *= (bad==0)
            sqrtwflux *= (bad==0)
            nout_iter += np.sum(bad)

        nout_tot += nout_iter

        sum_chi2=float(np.sum(chi2))
        ndf=int(np.sum(chi2>0)-nwave-nstds*2)
        chi2pdf=0.
        if ndf>0 :
            chi2pdf=sum_chi2/ndf

        # normalize to get a mean fiberflat=1
        mean=np.nanmean(smooth_fiber_correction,axis=0)
        smooth_fiber_correction /= mean

        log.info("iter #%d chi2=%f ndf=%d chi2pdf=%f nout=%d mean=%f"%(iteration,sum_chi2,ndf,chi2pdf,nout_iter,np.mean(mean)))

        if nout_iter == 0 and np.max(np.abs(mean-1))<0.005 :
            break

    # smooth_fiber_correction does not converge exactly to one on average, so we apply its mean to the calibration
    # (tested on sims)
    calibration /= mean

    log.info("nout tot=%d"%nout_tot)

    # solve once again to get deconvolved variance
    #calibration,calibcovar=cholesky_solve_and_invert(A.todense(),B)
    calibcovar=np.linalg.inv(A)
    calibvar=np.diagonal(calibcovar)
    log.info("mean(var)={0:f}".format(np.mean(calibvar)))

    calibvar=np.array(np.diagonal(calibcovar))
    # apply the mean (as in the iterative loop)
    calibvar *= mean**2
    calibivar=(calibvar>0)/(calibvar+(calibvar==0))

    # we also want to save the convolved calibration and a calibration variance
    # first compute average resolution
    mean_res_data=np.mean(frame.resolution_data,axis=0)
    R = Resolution(mean_res_data)
    # compute convolved calib
    ccalibration = np.zeros(frame.flux.shape)
    for i in range(frame.nspec):
        ccalibration[i]=frame.R[i].dot(calibration)/frame.R[i].dot(np.ones(calibration.shape))

    # Use diagonal of mean calibration covariance for output.
    ccalibcovar=R.dot(calibcovar).dot(R.T.todense())
    ccalibvar=np.array(np.diagonal(ccalibcovar))

    # apply the mean (as in the iterative loop)
    ccalibvar *= mean**2
    ccalibivar=(ccalibvar>0)/(ccalibvar+(ccalibvar==0))

    # convert to 2D
    # For now this is the same for all fibers; in the future it may not be
    ccalibivar = np.tile(ccalibivar, frame.nspec).reshape(frame.nspec, frame.nwave)

    # need to do better here
    mask = (ccalibivar==0).astype(np.int32)

    # return calibration, calibivar, mask, ccalibration, ccalibivar
    return FluxCalib(stdstars.wave, ccalibration, ccalibivar, mask, R.dot(calibration))\
        #, (sqrtwmodel, sqrtwflux, current_ivar, chi2)



class FluxCalib(object):
    def __init__(self, wave, calib, ivar, mask, meancalib=None):
        """Lightweight wrapper object for flux calibration vectors

        Args:
            wave : 1D[nwave] input wavelength (Angstroms)
            calib: 2D[nspec, nwave] calibration vectors for each spectrum
            ivar : 2D[nspec, nwave] inverse variance of calib
            mask : 2D[nspec, nwave] mask of calib (0=good)
            meancalib : 1D[nwave] mean convolved calibration (optional)

        All arguments become attributes, plus nspec,nwave = calib.shape

        The calib vector should be such that

            [1e-17 erg/s/cm^2/A] = [photons/A] / calib
        """
        assert wave.ndim == 1
        assert calib.ndim == 2
        assert calib.shape == ivar.shape
        assert calib.shape == mask.shape
        assert np.all(ivar >= 0)

        self.nspec, self.nwave = calib.shape
        self.wave = wave
        self.calib = calib
        self.ivar = ivar
        self.mask = util.mask32(mask)
        self.meancalib = meancalib

def apply_flux_calibration(frame, fluxcalib):
    """
    Applies flux calibration to input flux and ivar

    Args:
        frame: Spectra object with attributes wave, flux, ivar, resolution_data
        fluxcalib : FluxCalib object with wave, calib, ...

    Modifies frame.flux and frame.ivar
    """
    log=get_logger()
    log.info("starting")

    # check same wavelength, die if not the case
    mval=np.max(np.abs(frame.wave-fluxcalib.wave))
    if mval > 0.00001 :
        log.error("not same wavelength (should raise an error instead)")
        sys.exit(12)

    nwave=frame.nwave
    nfibers=frame.nspec

    """
    F'=F/C
    Var(F') = Var(F)/C**2 + F**2*(  d(1/C)/dC )**2*Var(C)
    = 1/(ivar(F)*C**2) + F**2*(1/C**2)**2*Var(C)
    = 1/(ivar(F)*C**2) + F**2*Var(C)/C**4
    = 1/(ivar(F)*C**2) + F**2/(ivar(C)*C**4)
    """
    # for fiber in range(nfibers) :
    #     C = fluxcalib.calib[fiber]
    #     flux[fiber]=frame.flux[fiber]*(C>0)/(C+(C==0))
    #     ivar[fiber]=(ivar[fiber]>0)*(civar[fiber]>0)*(C>0)/(   1./((ivar[fiber]+(ivar[fiber]==0))*(C**2+(C==0))) + flux[fiber]**2/(civar[fiber]*C**4+(civar[fiber]*(C==0)))   )

    C = fluxcalib.calib
    frame.flux = frame.flux * (C>0) / (C+(C==0))
    frame.ivar = (frame.ivar>0) * (fluxcalib.ivar>0) * (C>0) / (1./((frame.ivar+(frame.ivar==0))*(C**2+(C==0))) + frame.flux**2/(fluxcalib.ivar*C**4+(fluxcalib.ivar*(C==0)))   )


def ZP_from_calib(wave, calib):
    """ Calculate the ZP in AB magnitudes given the calibration and the wavelength arrays
    Args:
        wave:  1D array (A)
        calib:  1D array (converts erg/s/A to photons/s/A)

    Returns:
      ZP_AB: 1D array of ZP values in AB magnitudes

    """
    ZP_flambda = 1e-17 / calib  # erg/s/cm^2/A
    ZP_fnu = ZP_flambda * wave**2 / (2.9979e18)  # c in A/s
    # Avoid 0 values
    ZP_AB = np.zeros_like(ZP_fnu)
    gdZ = ZP_fnu > 0.
    ZP_AB[gdZ] = -2.5 * np.log10(ZP_fnu[gdZ]) - 48.6
    # Return
    return ZP_AB


def qa_fluxcalib(param, frame, fluxcalib, model_tuple):#, indiv_stars):
    """
    Args:
        param: dict of QA parameters
        frame: Frame
        fluxcalib: FluxCalib
        model_tuple : tuple of model data for standard stars (read from stdstars-...fits)

    Returns:
        qadict: dict of QA outputs
          Need to record simple Python objects for yaml (str, float, int)

    """
    log = get_logger()
    qadict = {}

    # Unpack model
    input_model_flux,input_model_wave,input_model_fibers=model_tuple

    # Standard stars
    stdfibers = (frame.fibermap['OBJTYPE'] == 'STD')
    stdstars = frame[stdfibers]
    nstds = np.sum(stdfibers)
    try:
        assert np.array_equal(frame.fibers[stdfibers], input_model_fibers)
    except AssertionError:
        log.error("Bad indexing in standard stars")

    # Calculate ZP for mean spectrum
    #medcalib = np.median(fluxcalib.calib,axis=0)
    medcalib = np.median(fluxcalib.calib[stdfibers],axis=0)
    ZP_AB = ZP_from_calib(fluxcalib.wave, medcalib)  # erg/s/cm^2/A

    # ZP at fiducial wavelength (AB mag for 1 photon/s/A)
    iZP = np.argmin(np.abs(fluxcalib.wave-param['ZP_WAVE']))
    qadict['ZP'] = float(np.median(ZP_AB[iZP-10:iZP+10]))

    # Unpack star data
    #sqrtwmodel, sqrtwflux, current_ivar, chi2 = indiv_stars

    # RMS
    qadict['NSTARS_FIBER'] = int(nstds)
    ZP_fiducial = np.zeros(nstds)
    for ii in range(nstds):
        # Model flux
        model_flux=resample_flux(stdstars.wave,input_model_wave,input_model_flux[ii])
        convolved_model_flux=stdstars.R[ii].dot(model_flux)
        # Good pixels
        gdp = stdstars.ivar[ii, :] > 0.
        icalib = stdstars.flux[ii, gdp] / convolved_model_flux[gdp]
        i_wave = fluxcalib.wave[gdp]
        ZP_stars = ZP_from_calib(i_wave, icalib)
        iZP = np.argmin(np.abs(i_wave-param['ZP_WAVE']))
        ZP_fiducial[ii] = float(np.median(ZP_stars[iZP-10:iZP+10]))
    qadict['RMS_ZP'] = float(np.std(ZP_fiducial))

    # MAX ZP Offset
    #stdfibers = np.where(frame.fibermap['OBJTYPE'] == 'STD')[0]
    ZPoffset = np.abs(ZP_fiducial-qadict['ZP'])
    qadict['MAX_ZP_OFF'] = [float(np.max(ZPoffset)),
                            int(stdfibers[np.argmax(ZPoffset)])]
    if qadict['MAX_ZP_OFF'] > param['MAX_ZP_OFF']:
        log.warn("Bad standard star ZP {:g}, in fiber {:d}".format(
                qadict['MAX_ZP_OFF'][0], qadict['MAX_ZP_OFF'][1]))
    # Return
    return qadict
