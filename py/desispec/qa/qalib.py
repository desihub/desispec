"""
simple low level library functions for offline and online qas
"""

import os
import yaml
import numpy as np
import scipy.stats
from scipy import optimize
from desiutil import stats as dustat
from desiutil.log import get_logger
from desispec.io.meta import findfile
from desispec.preproc import _parse_sec_keyword

log=get_logger()

def ampregion(image):
    """
    Get the pixel boundary regions for amps

    Args:
        image: desispec.image.Image object
    """
    pixboundary=[]
    for kk in ['1','2','3','4']: #- 4 amps
        #- get the amp region in pix
        ampboundary=_parse_sec_keyword(image.meta["CCDSEC"+kk])
        pixboundary.append(ampboundary)
    return pixboundary

def fiducialregion(frame,psf):
    """
    Get the fiducial amplifier regions on the CCD pixel to fiber by wavelength space

    Args:
        frame: desispec.frame.Frame object
        psf: desispec.psf.PSF like object
    """
    startspec=0 #- will be None if don't have fibers on the right of the CCD.
    endspec=499 #- will be None if don't have fibers on the right of the CCD
    startwave0=0 #- lower index for the starting fiber
    startwave1=0 #- lower index for the last fiber for the amp region
    endwave0=frame.wave.shape[0] #- upper index for the starting fiber
    endwave1=frame.wave.shape[0] #- upper index for the last fiber for that amp
    pixboundary=[]
    fidboundary=[]

    #- Adding the min, max boundary individually for the benefit of dumping to yaml.
    leftmax=499 #- for amp 1 and 3
    rightmin=0 #- for amp 2 and 4
    bottommax=frame.wave.shape[0] #- for amp 1 and 2
    topmin=0 #- for amp 3 and 4

    #- Loop over each amp
    for kk in ['1','2','3','4']: #- 4 amps
        #- get the amp region in pix
        ampboundary=_parse_sec_keyword(frame.meta["CCDSEC"+kk])
        pixboundary.append(ampboundary)
        for ispec in range(frame.flux.shape[0]):
            if np.all(psf.x(ispec) > ampboundary[1].start):
                startspec=ispec
                #-cutting off wavelenth boundaries from startspec
                yy=psf.y(ispec,frame.wave)
                k=np.where(yy > ampboundary[0].start)[0]
                startwave0=k[0]
                yy=psf.y(ispec,frame.wave)
                k=np.where(yy < ampboundary[0].stop)[0]
                endwave0=k[-1]
                break
            else:
                startspec=None
                startwave0=None
                endwave0=None
        if startspec is not None:
            for ispec in range(frame.flux.shape[0])[::-1]:
                if np.all(psf.x(ispec) < ampboundary[1].stop):
                    endspec=ispec
                    #-cutting off wavelenth boundaries from startspec
                    yy=psf.y(ispec,frame.wave)
                    k=np.where(yy > ampboundary[0].start)[0]
                    startwave1=k[0]
                    yy=psf.y(ispec,frame.wave)
                    k=np.where(yy < ampboundary[0].stop)[0]
                    endwave1=k[-1]
                    break
        else:
            endspec=None
            startwave1=None
            endwave1=None
        if startwave0 is not None and startwave1 is not None:
            startwave=max(startwave0,startwave1)
        else: startwave = None
        if endwave0 is not None and endwave1 is not None:
            endwave=min(endwave0,endwave1)
        else: endwave = None
        if endspec is not None:
            #endspec+=1 #- last entry exclusive in slice, so add 1
            #endwave+=1

            if endspec < leftmax:
                leftmax=endspec
            if startspec > rightmin:
                rightmin=startspec
            if endwave < bottommax:
                bottommax=endwave
            if startwave > topmin:
                topmin=startwave
        else:
            rightmin=0 #- Only if no spec in right side of CCD. passing 0 to encertain valid data type. Nontype throws a type error in yaml.dump.

        #fiducialb=(slice(startspec,endspec,None),slice(startwave,endwave,None))  #- Note: y,x --> spec, wavelength
        #fidboundary.append(fiducialb)

    #- return pixboundary,fidboundary
    return leftmax,rightmin,bottommax,topmin

def slice_fidboundary(frame,leftmax,rightmin,bottommax,topmin):
    """
    leftmax,rightmin,bottommax,topmin - Indices in spec-wavelength space for different amps (e.g output from fiducialregion function)
    #- This could be merged to fiducialregion function

    Returns (list):
        list of tuples of slices for spec- wavelength boundary for the amps.
    """
    leftmax+=1 #- last entry not counted in slice
    bottommax+=1
    if rightmin ==0:
        return [(slice(0,leftmax,None),slice(0,bottommax,None)), (slice(None,None,None),slice(None,None,None)),
                (slice(0,leftmax,None),slice(topmin,frame.wave.shape[0],None)),(slice(None,None,None),slice(None,None,None))]
    else:
        return [(slice(0,leftmax,None),slice(0,bottommax,None)), (slice(rightmin,frame.nspec,None),slice(0,bottommax,None)),
                (slice(0,leftmax,None),slice(topmin,frame.wave.shape[0],None)),(slice(rightmin,frame.nspec,None),slice(topmin,frame.wave.shape[0]-1,None))]


def getrms(image):
    """
    Calculate the rms of the pixel values)

    Args:
        image: 2d array
    """
    pixdata=image.ravel()
    rms=np.std(pixdata)
    return rms


def countpix(image,nsig=None):
    """
    Count the pixels above a given threshold in units of sigma.

    Args:
        image: 2d image array
        nsig: threshold in units of sigma, e.g 2 for 2 sigma
    """
    sig=np.std(image.ravel())
    counts_nsig=np.where(image.ravel() > nsig*sig)[0].shape[0]
    return counts_nsig

def countbins(flux,threshold=0):
    """
    Count the number of bins above a given threshold on each fiber

    Args:
        flux: 2d (nspec,nwave)
        threshold: threshold counts
    """
    counts=np.zeros(flux.shape[0])
    for ii in range(flux.shape[0]):
        ok=np.where(flux[ii]> threshold)[0]
        counts[ii]=ok.shape[0]
    return counts

def continuum(wave,flux,wmin=None,wmax=None):
    """
    Find the median continuum of the spectrum inside a wavelength region.

    Args:
        wave: 1d wavelength array
        flux: 1d counts/flux array
        wmin and wmax: region to consider for the continuum
    """
    if wmin is None:
        wmin=min(wave)
    if wmax is None:
        wmax=max(wave)

    kk=np.where((wave>wmin) & (wave < wmax))
    newwave=wave[kk]
    newflux=flux[kk]
    #- find the median continuum
    medcont=np.median(newflux)
    return medcont

def integrate_spec(wave,flux):
    """
    Calculate the integral of the spectrum in the given range using trapezoidal integration

    Note: limits of integration are min and max values of wavelength

    Args:
        wave: 1d wavelength array
        flux: 1d flux array
    """
    integral=np.trapz(flux,wave)
    return integral


def sky_continuum(frame, wrange1, wrange2):
    """ 
    QA Algorithm for sky continuum.

    To be called from desispec.sky.qa_skysub and
    desispec.qa.qa_quicklook.Sky_Continuum.run_qa
    
    Args:
        frame:
        wrange1:
        wrange2:

    Returns:
        skyfiber, contfiberlow, contfiberhigh, meancontfiber, skycont

    """
    #- get the skyfibers first
    skyfiber=np.where(frame.fibermap['OBJTYPE']=='SKY')[0]
    nspec_sky=skyfiber.shape[0]
    if isinstance(wrange1,list): # Offline list format
        wminlow,wmaxlow=wrange1
        wminhigh,wmaxhigh=wrange2
    else: # Quick look string format
        wminlow,wmaxlow=[float(w) for w in wrange1.split(',')]
        wminhigh,wmaxhigh=[float(w) for w in wrange2.split(',')]
    selectlow=np.where((frame.wave>wminlow) & (frame.wave<wmaxlow))[0]
    selecthigh=np.where((frame.wave>wminhigh) & (frame.wave < wmaxhigh))[0]

    contfiberlow=[]
    contfiberhigh=[]
    meancontfiber=[]
    for ii in skyfiber:
        contlow=continuum(frame.wave[selectlow],frame.flux[ii,selectlow])
        conthigh=continuum(frame.wave[selecthigh],frame.flux[ii,selecthigh])
        contfiberlow.append(contlow)
        contfiberhigh.append(conthigh)
        meancontfiber.append(np.mean((contlow,conthigh)))
    skycont=np.mean(meancontfiber) #- over the entire CCD (skyfibers)

    # Return
    return skyfiber, contfiberlow, contfiberhigh, meancontfiber, skycont


def sky_peaks(param, frame, dw=2, amps=False):

    # define sky peaks and wavelength region around peak flux to be integrated
    camera = frame.meta['CAMERA']
    peaks=np.array(param['{:s}_PEAKS'.format(camera[0].upper())])

    nspec_counts=[]
    sky_counts=[]
    nspec_counts_rms=[]
    amp1=[]
    amp2=[]
    amp3=[]
    amp4=[]
    rmsamp1=[]
    rmsamp2=[]
    rmsamp3=[]
    rmsamp4=[]
    for i in range(frame.flux.shape[0]):
        peak_fluxes = []
        for peak in peaks:
            iwave = np.argmin(np.abs(frame.wave-peak))
            peak_fluxes.append(np.trapz(frame.flux[i,iwave-dw:iwave+dw+1]))

        # Sum
        sum_counts=np.sum(peak_fluxes)/frame.meta["EXPTIME"]
        sum_counts_rms=np.sum(peak_fluxes)/np.sqrt(frame.meta["EXPTIME"])  # This looks funny to me..
        nspec_counts.append(sum_counts)
        nspec_counts_rms.append(sum_counts_rms)

        # Sky?
        if frame.fibermap['OBJTYPE'][i]=='SKY':
            sky_counts.append(sum_counts)
            '''
            if amps:
                if frame.fibermap['FIBER'][i]<240:
                    if camera[0]=="b":
                        amp1_flux=peak1_flux/frame.meta["EXPTIME"]
                        amp3_flux=np.sum((peak2_flux+peak3_flux)/frame.meta["EXPTIME"])
                        rmsamp1_flux=peak1_flux/np.sqrt(frame.meta["EXPTIME"])
                        rmsamp3_flux=np.sum((peak2_flux+peak3_flux)/np.sqrt(frame.meta["EXPTIME"]))
                    if camera[0]=="r":
                        amp1_flux=np.sum((peak1_flux+peak2_flux)/frame.meta["EXPTIME"])
                        amp3_flux=np.sum((peak3_flux+peak4_flux+peak5_flux)/frame.meta["EXPTIME"])
                        rmsamp1_flux=np.sum((peak1_flux+peak2_flux)/np.sqrt(frame.meta["EXPTIME"]))
                        rmsamp3_flux=np.sum((peak3_flux+peak4_flux+peak5_flux)/np.sqrt(frame.meta["EXPTIME"]))
                    if camera[0]=="z":
                        amp1_flux=np.sum((peak1_flux+peak2_flux+peak3_flux)/frame.meta["EXPTIME"])
                        amp3_flux=np.sum((peak4_flux+peak5_flux+peak6_flux)/frame.meta["EXPTIME"])
                        rmsamp1_flux=np.sum((peak1_flux+peak2_flux+peak3_flux)/np.sqrt(frame.meta["EXPTIME"]))
                        rmsamp3_flux=np.sum((peak4_flux+peak5_flux+peak6_flux)/np.sqrt(frame.meta["EXPTIME"]))
                    amp1.append(amp1_flux)
                    amp3.append(amp3_flux)
                    rmsamp1.append(rmsamp1_flux)
                    rmsamp3.append(rmsamp3_flux)
                if frame.fibermap['FIBER'][i]>260:
                    if camera[0]=="b":
                        amp2_flux=peak1_flux/frame.meta["EXPTIME"]
                        amp4_flux=np.sum((peak2_flux+peak3_flux)/frame.meta["EXPTIME"])
                        rmsamp2_flux=peak1_flux/np.sqrt(frame.meta["EXPTIME"])
                        rmsamp4_flux=np.sum((peak2_flux+peak3_flux)/np.sqrt(frame.meta["EXPTIME"]))
                    if camera[0]=="r":
                        amp2_flux=np.sum((peak1_flux+peak2_flux)/frame.meta["EXPTIME"])
                        amp4_flux=np.sum((peak3_flux+peak4_flux+peak5_flux)/frame.meta["EXPTIME"])
                        rmsamp2_flux=np.sum((peak1_flux+peak2_flux)/np.sqrt(frame.meta["EXPTIME"]))
                        rmsamp4_flux=np.sum((peak3_flux+peak4_flux+peak5_flux)/np.sqrt(frame.meta["EXPTIME"]))
                    if camera[0]=="z":
                        amp2_flux=np.sum((peak1_flux+peak2_flux+peak3_flux)/frame.meta["EXPTIME"])
                        amp4_flux=np.sum((peak4_flux+peak5_flux+peak6_flux)/frame.meta["EXPTIME"])
                        rmsamp2_flux=np.sum((peak1_flux+peak2_flux+peak3_flux)/np.sqrt(frame.meta["EXPTIME"]))
                        rmsamp4_flux=np.sum((peak4_flux+peak5_flux+peak6_flux)/np.sqrt(frame.meta["EXPTIME"]))
                    amp2.append(amp2_flux)
                    amp4.append(amp4_flux)
                    rmsamp2.append(rmsamp2_flux)
                    rmsamp4.append(rmsamp4_flux)
            '''

    nspec_counts = np.array(nspec_counts)
    sky_counts = np.array(sky_counts)
    # Return
    return nspec_counts, sky_counts


def sky_resid(param, frame, skymodel, quick_look=False):
    """ QA Algorithm for sky residual
    To be called from desispec.sky.qa_skysub and desispec.qa.qa_quicklook.Sky_residual.run_qa
    Args:
        param : dict of QA parameters
        frame : desispec.Frame object after sky subtraction
        skymodel : desispec.SkyModel object
    Returns a qa dictionary for sky resid
    """
    # Output dict
    qadict = {}
    qadict['NREJ'] = int(skymodel.nrej)

    if quick_look:
        qadict['RA'] = frame.fibermap['RA_TARGET']
        qadict['DEC'] = frame.fibermap['DEC_TARGET']

    # Grab sky fibers on this frame
    skyfibers = np.where(frame.fibermap['OBJTYPE'] == 'SKY')[0]
    assert np.max(skyfibers) < 500  #- indices, not fiber numbers
    nfibers=len(skyfibers)
    qadict['NSKY_FIB'] = int(nfibers)

    #current_ivar=frame.ivar[skyfibers].copy()
    #flux = frame.flux[skyfibers]

    # Record median flux
    qadict['MED_SKY'] = np.median(skymodel.flux[skyfibers])  # Counts

    #- Residuals
    res=frame.flux[skyfibers] #- as this frame is already sky subtracted
    res_ivar=frame.ivar[skyfibers]

    # Chi^2 and Probability
    chi2_fiber = np.sum(res_ivar*(res**2),1)
    chi2_prob = np.zeros(nfibers)
    for ii in range(nfibers):
        # Stats
        dof = np.sum(res_ivar[ii,:] > 0.)
        chi2_prob[ii] = scipy.stats.distributions.chi2.sf(chi2_fiber[ii], dof)
    # Bad models
    qadict['NBAD_PCHI'] = int(np.sum(chi2_prob < param['PCHI_RESID']))
    if qadict['NBAD_PCHI'] > 0:
        log.warning("Bad Sky Subtraction in {:d} fibers".format(
                qadict['NBAD_PCHI']))
    # Median residual
    qadict['MED_RESID'] = float(np.median(res)) # Median residual (counts)
    log.info("Median residual for sky fibers = {:g}".format(
        qadict['MED_RESID']))

    # Residual percentiles
    perc = dustat.perc(res, per=param['PER_RESID'])
    qadict['RESID_PER'] = [float(iperc) for iperc in perc]

    #SE: commented by popular demand!
    #qadict['RESID'] = []

    qadict["SKYFIBERID"]=skyfibers.tolist()
    #- Residuals in wave and fiber axes
    if quick_look:
        qadict["MED_RESID_WAVE"]=np.median(res,axis=0)
        qadict["MED_RESID_FIBER"]=np.median(res,axis=1)
        #- Weighted average for each bin on all fibers
        qadict["WAVG_RES_WAVE"]=np.zeros(res.shape[1])
        sw=np.sum(res_ivar,axis=0)
        qadict["WAVG_RES_WAVE"][sw>0] = np.sum(res*res_ivar,axis=0)[sw>0] / sw[sw>0]
        
    #- Histograms for residual/sigma #- inherited from qa_plots.frame_skyres()
    if quick_look:
        binsz = param['BIN_SZ']
        gd_res = res_ivar > 0.
        devs = res[gd_res] * np.sqrt(res_ivar[gd_res])
        i0, i1 = int( np.min(devs) / binsz) - 1, int( np.max(devs) / binsz) + 1
        rng = tuple( binsz*np.array([i0,i1]) )
        nbin = i1-i0
        hist, edges = np.histogram(devs, range=rng, bins=nbin)

        #SE: commented this because didn't seem to be needed to be saved in the dictionary
        #qadict['DEVS_1D'] = hist.tolist() #- histograms for deviates
        #qadict['DEVS_EDGES'] = edges.tolist() #- Bin edges

    #- Add additional metrics for quicklook
    if quick_look:
        qadict["WAVELENGTH"]=frame.wave
    # Return
    return qadict


def SN_ratio(flux,ivar):
    """
    SN Ratio
    median snr for the spectra, flux should be sky subtracted.

    Args:
        flux (array): 2d [nspec,nwave] the signal (typically for spectra,
            this comes from frame object
        ivar (array): 2d [nspec,nwave] corresponding inverse variance

    Returns:
        medsnr (array): 1d [nspec]
    """
    #- we calculate median and total S/N assuming no correlation bin by bin
    snr = flux * np.sqrt(ivar)
    medsnr = np.median(snr, axis=1)
    return medsnr #, totsnr


def SignalVsNoise(frame,params,fidboundary=None):
    """
    Signal vs. Noise

    Take flux and inverse variance arrays and calculate S/N for individual
    targets (ELG, LRG, QSO, STD) and for each amplifier of the camera.

    Args:
        flux (array): 2d [nspec,nwave] the signal (typically for spectra,
            this comes from frame object
        ivar (array): 2d [nspec,nwave] corresponding inverse variance
        fidboundary : list of slices indicating where to select in fiber
            and wavelength directions for each amp (output of slice_fidboundary function)
    """
    thisfilter='DECAM_R' #- should probably come from param. Hard coding for now
    mags=frame.fibermap['MAG']
    filters=frame.fibermap['FILTER']

    medsnr=SN_ratio(frame.flux,frame.ivar)

    #- Calculate median SNR per bin and associate with imaging Mag. for ELG fibers
    elgfibers=np.where(frame.fibermap['OBJTYPE']=='ELG')[0]
    elg_medsnr=medsnr[elgfibers]
    elg_mag=np.zeros(len(elgfibers))
    for ii,fib in enumerate(elgfibers):
        if thisfilter not in filters[fib]:
            #- raise ValueError("{} is not available filter for fiber {}".format(thisfilter,fib))
            print("WARNING!!! {} is not available filter for fiber {}".format(thisfilter,fib))
            elg_mag[ii]=None
        else:
            elg_mag[ii]=mags[fib][filters[fib]==thisfilter]
    elg_snr_mag=np.array((elg_medsnr,elg_mag)) #- not storing fiber number

    #- Calculate median SNR, associate with imaging Mag for LRGs
    lrgfibers=np.where(frame.fibermap['OBJTYPE']=='LRG')[0]
    lrg_medsnr=medsnr[lrgfibers]
    lrg_mag=np.zeros(len(lrgfibers))
    for ii,fib in enumerate(lrgfibers):
        if thisfilter not in filters[fib]:
            print("WARNING!!! {} is not available filter for fiber {}".format(thisfilter,fib))
            lrg_mag[ii]=None
        else:
            lrg_mag[ii]=mags[fib][filters[fib]==thisfilter]
    lrg_snr_mag=np.array((lrg_medsnr,lrg_mag))

    #- Calculate median SNR, associate with imaging Mag. for QSOs
    qsofibers=np.where(frame.fibermap['OBJTYPE']=='QSO')[0]
    qso_medsnr=medsnr[qsofibers]
    qso_mag=np.zeros(len(qsofibers))
    for ii,fib in enumerate(qsofibers):
        if thisfilter not in filters[fib]:
            print("WARNING!!! {} is not available filter for fiber {}".format(thisfilter,fib))
            qso_mag[ii]=None
        else:
            qso_mag[ii]=mags[fib][filters[fib]==thisfilter]
    qso_snr_mag=np.array((qso_medsnr,qso_mag))

    #- Calculate median SNR, associate with Mag. for STD stars
    stdfibers=np.where(frame.fibermap['OBJTYPE']=='STD')[0]
    std_medsnr=medsnr[stdfibers]
    std_mag=np.zeros(len(stdfibers))
    for ii,fib in enumerate(stdfibers):
        if thisfilter not in filters[fib]:
            print("WARNING!!! {} is not available filter for fiber {}".format(thisfilter,fib))
            std_mag[ii]=None
        else:
            std_mag[ii]=mags[fib][filters[fib]==thisfilter]
    std_snr_mag=np.array((std_medsnr,std_mag))

    #- Median S/N for different amp zones.
    average_amp = None
    if fidboundary is not None:
        averages=[]
        for ii in range(4):
            if fidboundary[ii][0].start is not None:  #- have fibers in this amp?
                medsnramp=SN_ratio(frame.flux[fidboundary[ii]],frame.ivar[fidboundary[ii]])
                averages.append(np.mean(medsnramp))
            else:
                averages.append(None)

        average_amp=np.array(averages)

    elg_fidmag_snr = []
    star_fidmag_snr = []

    ra = frame.fibermap['RA_TARGET']
    dec = frame.fibermap['DEC_TARGET']

    #- fill QA dict with metrics:
    qadict={"RA":ra, "DEC":dec, "MEDIAN_SNR":medsnr,"MEDIAN_AMP_SNR":average_amp, "ELG_FIBERID":elgfibers.tolist(), "ELG_SNR_MAG": elg_snr_mag, "LRG_FIBERID":lrgfibers.tolist(), "LRG_SNR_MAG": lrg_snr_mag, "QSO_FIBERID": qsofibers.tolist(), "QSO_SNR_MAG": qso_snr_mag, "STAR_FIBERID": stdfibers.tolist(), "STAR_SNR_MAG":std_snr_mag, "ELG_FIDMAG_SNR":elg_fidmag_snr, "STAR_FIDMAG_SNR":star_fidmag_snr}

    return qadict

def s2n_funcs(exptime=None, r2=0.):
    """
    Functions for fitting S/N

    Args:
        exptime: float, optional
        r2: float, optional  -- RN^2

    Returns:
        funcMap: dict

    """
    funcMap={"linear":lambda x,a,b:a+b*x,
             "poly":lambda x,a,b,c:a+b*x+c*x**2,
             "astro":lambda x,a,b:(exptime*a*x)/np.sqrt(exptime*(a*x+b)+r2)
             }
    return funcMap

def SNRFit(frame,night,camera,expid,objlist,params,fidboundary=None):
    """
    Signal vs. Noise With fitting

    Take flux and inverse variance arrays and calculate S/N for individual
    targets (ELG, LRG, QSO, STD) and for each amplifier of the camera.
    then fit the log(snr)=a+b*mag or log(snr)=poly(mag)
    
    see http://arXiv.org/abs/0706.1062v2 for proper fitting of power-law distributions
    it is not implemented here!

    qadict has the following data model
      "MAGNITUDES" : ndarray - Depends on camera (DECAM_G, DECAM_R, DECAM_Z)
      "MEDIAN_SNR" : ndarray (nfiber)
      "NUM_NEGATIVE_SNR" : int
      "SNR_MAG_TGT"
      "FITCOEFF_TGT" : list
      "FITCOVAR_TGT" : list
      "SNR_RESID" : list
      "FIDSNR_TGT"
      "RA" : ndarray (nfiber)
      "DEC" : ndarray (nfiber)
      "OBJLIST" : list - Save a copy to make sense of the list order later
      "EXPTIME" : float
      "FIT_FILTER" : str
      "FILTERS" : list
      "r2" : float - Fitting parameter

    Args:
        frame: desispec.Frame object
        night :
        camera :
        expid : int
        params: parameters dictionary
        {
          "Func": "linear", # Fit function type one of ["linear","poly"]
          "FIDMAG": 22.0, # magnitude to evaluate the fit
          "Filter":"DECAM_R", #filter name
        }

        fidboundary : list of slices indicating where to select in fiber
            and wavelength directions for each amp (output of slice_fidboundary function)

    Returns:
        qadict : dict
    """

    #- Get imaging magnitudes and calculate SNR
    magnitudes=frame.fibermap['MAG']
    fmag=22.0
    if "FIDMAG" in params:
        fmag=params["FIDMAG"]
    filters=frame.fibermap['FILTER']
    mediansnr=SN_ratio(frame.flux,frame.ivar)
    qadict={"MEDIAN_SNR":mediansnr}
    qadict["FILTERS"] = filters[0,:]
    exptime=frame.meta["EXPTIME"]

    if camera[0] == 'b':
        thisfilter='DECAM_G' #- should probably come from param. Hard coding for now
    elif camera[0] =='r':
        thisfilter='DECAM_R'
    else:
        thisfilter='DECAM_Z'
    if "Filter" in params:
        thisfilter=params["Filter"]
    qadict["FIT_FILTER"] = thisfilter
    qadict["EXPTIME"] = exptime

    mag_filters=[]
    for mag in range(magnitudes.shape[0]):
        mag_filters.append([magnitudes[mag][0],magnitudes[mag][1],magnitudes[mag][2]])
    qadict["MAGNITUDES"]=mag_filters

    qadict["OBJLIST"]=list(objlist)

    #- Set up fit of SNR vs. Magnitude
    try:
        #- Get read noise from Get_RMS TODO: use header information for this
        rfile=findfile('ql_getrms_file',int(night),int(expid),camera,specprod_dir=os.environ['QL_SPEC_REDUX'])
        with open(rfile) as rf:
            rmsfile=yaml.load(rf)
        rmsval=rmsfile["METRICS"]["NOISE"]
        #- The factor of 1e-3 is a very basic (and temporary!!) flux calibration
        #- used to convert read noise to proper flux units
        r2=1e-3*rmsval**2
    except:
        r2 = 0.
        log.info("Was not able to obtain read noise from prior knowledge, fitting B+R**2...")

    #- Using astronomical SNR equation, fitting 'a'(throughput) and 'B'(sky background)
    #- If read noise is not available, fit 'a' and 'B+R**2'
    funcMap = s2n_funcs(exptime=exptime, r2=r2)

    fitfunc=funcMap["astro"]
    initialParams=[0.1,0.1]
    qadict["r2"]=r2

    neg_snr_tot=[]
    #- neg_snr_tot counts the number of times a fiber has a negative median SNR.  This should 
    #- not happen for non-sky fibers with actual flux in them.  However, it does happen rarely 
    #- in sims.  To avoid this, we omit such fibers in the fit, but keep count for diagnostic 
    #- purposes.

    #- Loop over each target type, and associate SNR and image magnitudes for each type.
    resid_snr=[]
    fidsnr_tgt=[]
    fitcoeff=[]
    fitcovar=[]
    snrmag=[]
    badfibs=[]
    fitsnr=[]
    for T in objlist:
        fibers=np.where(frame.fibermap['OBJTYPE']==T)[0]
        medsnr=mediansnr[fibers]
        mags=np.zeros(medsnr.shape)
        if T=="STD":# this should be fixed "STAR" or "STD" should be used consistently everywhere!
            T="STAR"
        for ii,fib in enumerate(fibers):
            if thisfilter not in filters[fib]:
                log.warning("WARNING!!! {} magnitude is not available filter for fiber {}".format(thisfilter,fib))
                mags[ii]=None
            else:
                mags[ii]=magnitudes[fib][filters[fib]==thisfilter]

        try:
	    #- Determine invalid SNR and mag values and remove
            m=mags
            s=medsnr
            neg_snr=len(np.where(medsnr<=0.0)[0])
            neg_snr_tot.append(neg_snr)
            neg_val=np.where(medsnr<=0.0)[0]
            inf_mag=np.where(m == np.inf)[0]
            nan_mag=np.where(np.isnan(m))[0]
            none_mag=np.where(m == np.array(None))[0]
            cut=[]
            cuts=[neg_val,inf_mag,nan_mag,none_mag]
            for cc in range(len(cuts)):
                if len(cuts[cc]) > 0:
                    for ci in range(len(cuts[cc])):
                        cut.append(cuts[cc][ci])
            if len(cut) > 0:
                m=list(m)
                s=list(s)
                makecut=sorted(list(set(cut)))
                for nn in range(len(makecut)):
                    m.remove(m[makecut[nn]])
                    s.remove(s[makecut[nn]])
                    for ni in range(len(makecut)):
                        makecut[ni]-=1
                m=np.array(m)
                s=np.array(s)
                log.warning("In fit of {}, had to remove NANs from data for fitting!".format(T))
            badfibs.append(fibers[sorted(list(set(cut)))])
            xs=m.argsort()
            #- Convert magnitudes to flux
            x=10**(-0.4*(m[xs]-22.5))
            med_snr=s[xs]
            y=med_snr
	    #- Fit SNR vs. Magnitude, evaluate at fiducial magnitude, 
            #- and store results in METRICS
            out=optimize.curve_fit(fitfunc,x,y,p0=initialParams)
            vs=list(out[0])
            cov=list(out[1])
            fitcoeff.append(vs)
            fitcovar.append(cov)
            fidsnr_tgt.append(fitfunc(fmag,*out[0]))
        except RuntimeError:
            log.warning("In fit of {}, Fit minimization failed!".format(T))
            vs=np.array(initialParams)
            vs.fill(np.nan)
            cov=np.empty((len(initialParams),len(initialParams)))
            cov.fill(np.nan)
            vs=list(vs)
            cov=list(cov)
            fitcoeff.append(vs)
            fitcovar.append(cov)
            fidsnr_tgt.append(np.nan)
            
        except scipy.optimize.OptimizeWarning:
            log.warning("WARNING!!! {} Covariance estimation failed!".format(T))
            vs=out[0]
            cov=np.empty((len(initialParams),len(initialParams)))
            cov.fill(np.nan)
            vs=list(vs)
            cov=list(cov)
            fitcoeff.append(vs)
            fitcovar.append(cov)
            fidsnr_tgt.append(np.nan)

        qadict["{:s}_FIBERID".format(T)]=fibers.tolist()
        snr_mag=[medsnr,mags]
        snrmag.append(snr_mag)

        #- Calculate residual SNR for focal plane plots
        #for mm in range(len(x)):
        #    snr = fitfunc(x[mm],*vs)
        #    fit_snr.append(snr)
        fit_snr = fitfunc(x, *vs)
        fitsnr += fit_snr.tolist()
        #for rr in range(len(fit_snr)):
        #    resid = (med_snr[rr] - fit_snr[rr]) / fit_snr[rr]
        #    resid_snr.append(resid)
        resid = (med_snr-fit_snr)/fit_snr
        resid_snr += resid.tolist()

    qadict["NUM_NEGATIVE_SNR"]=sum(neg_snr_tot)
    qadict["SNR_MAG_TGT"]=snrmag
    qadict["FITCOEFF_TGT"]=fitcoeff
    qadict["FITCOVAR_TGT"]=fitcovar
    qadict["SNR_RESID"]=resid_snr
    qadict["FIDSNR_TGT"]=fidsnr_tgt
    qadict["RA"]=frame.fibermap['RA_TARGET']
    qadict["DEC"]=frame.fibermap['DEC_TARGET']

    return qadict,badfibs,fitsnr

def gauss(x,a,mu,sigma):
    """
    Gaussian fit of input data
    """
    return a*np.exp(-(x-mu)**2/(2*sigma**2))
