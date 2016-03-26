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
import scipy, scipy.sparse, scipy.ndimage
import sys
from astropy import units

#debug
#import pylab

#rebin spectra into new wavebins. This should be equivalent to desispec.interpolation.resample_flux. So may not be needed here
#But should move from here anyway.

def rebinSpectra(spectra,oldWaveBins,newWaveBins):
    tck=scipy.interpolate.splrep(oldWaveBins,spectra,s=0,k=5)
    specnew=scipy.interpolate.splev(newWaveBins,tck,der=0)
    return specnew

#import some global constants
import scipy.constants as const
h=const.h
pi=const.pi
e=const.e
c=const.c
erg=const.erg
hc= h/erg*c*1.e10 #(in units of ergsA)

def match_templates(wave, flux, ivar, resolution_data, stdwave, stdflux):
    """For each input spectrum, identify which standard star template is the closest
    match, factoring out broadband throughput/calibration differences.

    Args:
        wave : A dictionary of 1D array of vacuum wavelengths [Angstroms]. Example below.
        flux : A dictionary of 1D observed flux for the star
        ivar : A dictionary 1D inverse variance of flux
        resolution_data: resolution corresponding to the star's fiber
        stdwave : 1D standard star template wavelengths [Angstroms]
        stdflux : 2D[nstd, nwave] template flux

    Returns:
        stdflux[nspec, nwave] : standard star flux sampled at input wave
        stdindices[nspec] : indices of input standards for each match

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

    def applySmoothingFilter(flux):
        return scipy.ndimage.filters.median_filter(flux,200) # bin range has to be optimized


    rnorm=flux["r"]/applySmoothingFilter(flux["r"])
    bnorm=flux["b"]/applySmoothingFilter(flux["b"])
    znorm=flux["z"]/applySmoothingFilter(flux["z"])


   # propagate this normalization to ivar

    bivar=ivar["b"]*(applySmoothingFilter(flux["b"]))**2
    rivar=ivar["r"]*(applySmoothingFilter(flux["r"]))**2
    zivar=ivar["z"]*(applySmoothingFilter(flux["z"]))**2

    Chisq=1e100
    bestId=-1
    bchisq=0
    rchisq=0
    zchisq=0

    bmodels={}
    rmodels={}
    zmodels={}
    for i,v in enumerate(stdflux):
        bmodels[i]=rebinSpectra(v,stdwave,wave["b"])
        rmodels[i]=rebinSpectra(v,stdwave,wave["r"])
        zmodels[i]=rebinSpectra(v,stdwave,wave["z"])

    Models={"b":bmodels,"r":rmodels,"z":zmodels}

    def convolveModel(wave,resolution,flux):

        diags=np.arange(10,-11,-1)
        nwave=len(wave)
        convolved=np.zeros(nwave)
        R=Resolution(resolution)
        convolved=R.dot(flux)

        return convolved

    nstd=stdflux.shape[0]
    nstdwave=stdwave.shape[0]
    maxDelta=1e100
    bestId=-1
    red_Chisq=-1.

    for i in range(nstd):

        bconvolveFlux=convolveModel(wave["b"],resolution_data["b"],Models["b"][i])
        rconvolveFlux=convolveModel(wave["r"],resolution_data["r"],Models["r"][i])
        zconvolveFlux=convolveModel(wave["z"],resolution_data["z"],Models["z"][i])

        b_models=bconvolveFlux/applySmoothingFilter(bconvolveFlux)
        r_models=rconvolveFlux/applySmoothingFilter(rconvolveFlux)
        z_models=zconvolveFlux/applySmoothingFilter(zconvolveFlux)

        rdelta=np.sum(((r_models-rnorm)**2)*rivar)
        bdelta=np.sum(((b_models-bnorm)**2)*bivar)
        zdelta=np.sum(((z_models-znorm)**2)*zivar)
        if (rdelta+bdelta+zdelta)<maxDelta:
                bestmodel={"r":r_models,"b":b_models,"z":z_models}
                bestId=i
                maxDelta=(rdelta+bdelta+zdelta)
                dof=len(wave["b"])+len(wave["r"])+len(wave["z"])
                red_Chisq=maxDelta/dof

    return bestId,stdwave,stdflux[bestId],red_Chisq
    #Should we skip those stars with very bad Chisq?


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


def compute_flux_calibration(frame, input_model_wave,input_model_flux,nsig_clipping=4.):
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
    for fiber in range(model_flux.shape[0]) :
        model_flux[fiber]=resample_flux(stdstars.wave,input_model_wave,input_model_flux[fiber])

        # debug
        # pylab.plot(input_model_wave,input_model_flux[fiber])
        # pylab.plot(wave,model_flux[fiber],c="g")

        model_flux[fiber]=stdstars.R[fiber].dot(model_flux[fiber])

        # debug
        # pylab.plot(wave,model_flux[fiber],c="r")
        # pylab.show()

    # iterative fitting and clipping to get precise mean spectrum
    current_ivar=stdstars.ivar.copy()

    #- DEBUG HACK
    # ii = (current_ivar == 0)
    # log.warning('{} ivar=0 values'.format(np.count_nonzero(ii)))
    # current_ivar[ii] = np.median(current_ivar[~ii])
    #- DEBUG HACK

    smooth_fiber_correction=np.ones((stdstars.flux.shape))
    chi2=np.zeros((stdstars.flux.shape))


    sqrtwmodel=np.sqrt(current_ivar)*model_flux
    sqrtwflux=np.sqrt(current_ivar)*stdstars.flux


    # test
    # nstds=20
    nout_tot=0
    for iteration in range(20) :

        # fit mean calibration
        A=scipy.sparse.lil_matrix((nwave,nwave)).tocsr()
        B=np.zeros((nwave))

        # diagonal sparse matrix with content = sqrt(ivar)*flat of a given fiber
        SD=scipy.sparse.lil_matrix((nwave,nwave))

        # loop on fiber to handle resolution
        for fiber in range(nstds) :
            if fiber%10==0 :
                log.info("iter %d fiber %d"%(iteration,fiber))

            R = stdstars.R[fiber]

            # diagonal sparse matrix with content = sqrt(ivar)*flat
            SD.setdiag(sqrtwmodel[fiber])

            sqrtwmodelR = SD*R # each row r of R is multiplied by sqrtwmodel[r]

            A = A+(sqrtwmodelR.T*sqrtwmodelR).tocsr()
            B += sqrtwmodelR.T*sqrtwflux[fiber]

        log.info("iter %d solving"%iteration)

        calibration=cholesky_solve(A.todense(),B)

        log.info("iter %d fit smooth correction per fiber"%iteration)
        # fit smooth fiberflat and compute chi2
        smoothing_res=1000. #A

        for fiber in range(nstds) :
            if fiber%10==0 :
                log.info("iter %d fiber %d(smooth)"%(iteration,fiber))

            R = stdstars.R[fiber]

            #M = np.array(np.dot(R.todense(),mean_spectrum)).flatten()
            M = R.dot(calibration)*model_flux[fiber]

            #debug
            #pylab.plot(wave,flux[fiber],c="b")
            #pylab.plot(wave,M,c="r")
            #pylab.show()
            #continue

            F = stdstars.flux[fiber]/(M+(M==0))
            smooth_fiber_correction[fiber]=spline_fit(stdstars.wave,stdstars.wave,F,smoothing_res,current_ivar[fiber]*(M!=0))
            chi2[fiber]=current_ivar[fiber]*(stdstars.flux[fiber]-smooth_fiber_correction[fiber]*M)**2

            #pylab.plot(wave,F)
            #pylab.plot(wave,smooth_fiber_correction[fiber])


        #pylab.show()
        #sys.exit(12)


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
        ndf=int(np.sum(chi2>0)-nwave-nstds*(nwave/smoothing_res))
        chi2pdf=0.
        if ndf>0 :
            chi2pdf=sum_chi2/ndf

        # normalize to get a mean fiberflat=1
        mean=np.mean(smooth_fiber_correction,axis=0)
        sv_smooth = smooth_fiber_correction
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
    log.info("mean(var)={0:f}".format(np.mean(calibvar)))

    calibvar=np.array(np.diagonal(calibcovar))
    # apply the mean (as in the iterative loop)
    calibvar *= mean**2
    calibivar=(calibvar>0)/(calibvar+(calibvar==0))

    # we also want to save the convolved calibration and calibration variance
    # first compute average resolution
    mean_res_data=np.mean(frame.resolution_data,axis=0)
    R = Resolution(mean_res_data)
    # compute convolved calib
    ccalibration = np.zeros(frame.flux.shape)
    for i in range(frame.nspec):
        ccalibration[i]=frame.R[i].dot(calibration)

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
    mask=(ccalibivar==0).astype(int)

    # return calibration, calibivar, mask, ccalibration, ccalibivar
    return FluxCalib(stdstars.wave, ccalibration, ccalibivar, mask, R.dot(calibration)), (
        sqrtwmodel, sqrtwflux, current_ivar, chi2)



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
        
            [erg/s/cm^2/A] = [photons/A] / calib
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
        self.mask = mask
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
    ZP_flambda = 1. / calib  # erg/s/cm^2/A
    ZP_fnu = ZP_flambda * wave**2 / (2.9979e18)  # c in A/s
    ZP_AB = -2.5 * np.log10(ZP_fnu) - 48.6
    # Return
    return ZP_AB

def qa_fluxcalib(param, frame, fluxcalib, indiv_stars):
    """
    Args:
        param: dict of QA parameters
        frame: Frame
        fluxcalib: FluxCalib
        indiv_stars : tuple of data on individual star fibers

    Returns:
        qadict: dict of QA outputs
          Need to record simple Python objects for yaml (str, float, int)

    """
    log = get_logger()

    qadict = {}

    # Calculate ZP for mean spectrum
    ZP_AB = ZP_from_calib(fluxcalib.wave, fluxcalib.meancalib)  # erg/s/cm^2/A

    # ZP at fiducial wavelength (AB mag for 1 photon/s/A)
    iZP = np.argmin(np.abs(fluxcalib.wave-param['ZP_WAVE']))
    qadict['ZP'] = float(np.median(ZP_AB[iZP-10:iZP+10]))

    # Unpack star data
    sqrtwmodel, sqrtwflux, current_ivar, chi2 = indiv_stars

    # RMS
    nstars = sqrtwflux.shape[0]
    qadict['NSTARS_FIBER'] = int(nstars)
    ZP_stars = np.zeros_like(sqrtwflux)
    ZP_fiducial = np.zeros(nstars)
    for ii in range(nstars):
        # Good pixels
        gdp = current_ivar[ii, :] > 0.
        icalib = sqrtwflux[ii, gdp] / sqrtwmodel[ii, gdp]
        i_wave = fluxcalib.wave[gdp]
        ZP_stars = ZP_from_calib(i_wave, icalib)
        iZP = np.argmin(np.abs(i_wave-param['ZP_WAVE']))
        ZP_fiducial[ii] = float(np.median(ZP_stars[iZP-10:iZP+10]))
    qadict['RMS_ZP'] = float(np.std(ZP_fiducial))

    # MAX ZP Offset
    stdfibers = np.where(frame.fibermap['OBJTYPE'] == 'STD')[0]
    ZPoffset = np.abs(ZP_fiducial-qadict['ZP'])
    qadict['MAX_ZP_OFF'] = [float(np.max(ZPoffset)),
                            int(stdfibers[np.argmax(ZPoffset)])]
    if qadict['MAX_ZP_OFF'] > param['MAX_ZP_OFF']:
        log.warn("Bad standard star ZP {:g}, in fiber {:d}".format(
                qadict['MAX_ZP_OFF'][0], qadict['MAX_ZP_OFF'][1]))

    # Return
    return qadict

