"""
desispec.fluxcalibration
========================

Flux calibration routines.
"""

import numpy as np
from desispec.resolution import Resolution
from desispec.linalg import cholesky_solve
from desispec.linalg import cholesky_solve_and_invert
from desispec.linalg import spline_fit
from desispec.interpolation import resample_flux
from desispec.log import get_logger
from desispec.io.fluxcalibration import read_filter_response
from desispec.resolution import Resolution
import scipy,scipy.sparse, scipy.ndimage
import sys
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
        #print 'resolution',resolution[1].shape
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
        #print i, (rdelta+bdelta+zdelta)/(len(bwave)+len(rwave)+len(zwave))
        if (rdelta+bdelta+zdelta)<maxDelta:
                bestmodel={"r":r_models,"b":b_models,"z":z_models}
                bestId=i
                maxDelta=(rdelta+bdelta+zdelta)
                dof=len(wave["b"])+len(wave["r"])+len(wave["z"])
                red_Chisq=maxDelta/dof

    return bestId,stdwave,stdflux[bestId],red_Chisq
    #Should we skip those stars with very bad Chisq?


def normalize_templates(stdwave, stdflux, mags, filters, basepath):
    """Returns spectra normalized to input magnitudes.

    Args:
        stdwave : 1D array of standard star wavelengths [Angstroms]
        stdflux : 1D observed flux
        mags : 1D array of observed AB magnitudes
        filters : list of filter names for mags, e.g. ['SDSS_r', 'DECAM_g', ...]

    Only SDSS_r band is assumed to be used for normalization for now.
    """

    def ergs2photons(flux,wave):
        return flux*wave/hc

    def findappMag(flux,wave,filt):


        flux_in_photons=ergs2photons(flux,wave)
        flux_filt_integrated=np.dot(flux_in_photons,filt)

        ab_spectrum = 2.99792458 * 10**(18-48.6/2.5)/hc/wave #in photons/cm^2/s/A, taken from specex_flux_calibration.py)
        # Does this relation hold for all SDSS filters or there is some relative zero point adjustment? What about other filters?
        ab_spectrum_filt_integrated=np.dot(ab_spectrum,filt)

        if flux_filt_integrated <=0:
           appMag=99.
        else:
           appMag=-2.5*np.log10(flux_filt_integrated/ab_spectrum_filt_integrated)
        return appMag

    nstdwave=stdwave.size
    normflux=np.array(nstdwave)

    for i,v in enumerate(filters):
        #Normalizing using only SDSS_R band magnitude
        if v=='SDSS_R':
            refmag=mags[i]
            filter_response=read_filter_response(v,basepath) # outputs wavelength,qe
            rebinned_model_flux=rebinSpectra(stdflux,stdwave,filter_response[0])
            apMag=findappMag(rebinned_model_flux,filter_response[0],filter_response[1])
            print 'scaling SDSS_r mag',apMag,'to',refmag
            scalefac=10**((apMag-refmag)/2.5)
            normflux=stdflux*scalefac

    return stdwave,normflux

def convolveFlux(wave,resolution,flux):
    """I am writing this full convolution only for sky subtraction. It will be applied to sky model.
    """
    diags=np.arange(10,-11,-1)
    nwave=len(wave)
    nspec=resolution.shape[0]
    convolved=np.zeros((nspec,nwave))
    for i in range(nspec):
       R=Resolution(resolution[i])
       convolved[i]=R.dot(flux)

    return convolved


def compute_flux_calibration(wave,flux,ivar,resolution_data,input_model_wave,input_model_flux,nsig_clipping=4.):
    """Compute average frame throughtput based on data (wave,flux,ivar,resolution_data)
    and spectro-photometrically calibrated stellar models (model_wave,model_flux).
    Wave and model_wave are not necessarily on the same grid

    Input flux and model fiber indices have to match.

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
        model_flux[fiber]=resample_flux(wave,input_model_wave,input_model_flux[fiber])

        # debug
        # pylab.plot(input_model_wave,input_model_flux[fiber])
        # pylab.plot(wave,model_flux[fiber],c="g")

        R = Resolution(resolution_data[fiber])
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
            R = Resolution(resolution_data[fiber])

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

            R = Resolution(resolution_data[fiber])

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
    R = Resolution(mean_res_data)
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

def apply_flux_calibration(flux,ivar,resolution_data,wave,calibration,civar,cmask,cwave):
    """
    Applies flux calibration to input flux and ivar
    
    Args:
        flux : input flux[nspec, nwave] -- WILL BE MODIFIED IN-PLACE
        ivar : input ivar[nspec, nwave] -- WILL BE MODIFIED IN-PLACE
        resolution_data : 3D[nspec, ndiag, nwave] sparse resolution matrix data
        wave : 1D[nwave] wavelength of flux
        calibration, civar, cmask, cwave : from compute_flux_calibration()
    """
    log=get_logger()
    log.info("starting")

    # check same wavelength, die if not the case
    mval=np.max(np.abs(wave-cwave))
    if mval > 0.00001 :
        log.error("not same wavelength (should raise an error instead)")
        sys.exit(12)

    nwave=wave.size
    nfibers=flux.shape[0]

    for fiber in range(nfibers) :

        R = Resolution(resolution_data[fiber])
        C = R.dot(calibration)

        """
        F'=F/C
        Var(F') = Var(F)/C**2 + F**2*(  d(1/C)/dC )**2*Var(C)
        = 1/(ivar(F)*C**2) + F**2*(1/C**2)**2*Var(C)
        = 1/(ivar(F)*C**2) + F**2*Var(C)/C**4
        = 1/(ivar(F)*C**2) + F**2/(ivar(C)*C**4)
        """

        flux[fiber]=flux[fiber]*(C>0)/(C+(C==0))
        ivar[fiber]=(ivar[fiber]>0)*(civar[fiber]>0)*(C>0)/(   1./((ivar[fiber]+(ivar[fiber]==0))*(C**2+(C==0))) + flux[fiber]**2/(civar[fiber]*C**4+(civar[fiber]*(C==0)))   )
