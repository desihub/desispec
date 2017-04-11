"""
simple low level library functions for offline and online qas
"""

import numpy as np
import scipy.stats
from desiutil import stats as dustat
from desiutil.log import get_logger
log=get_logger()

def ampregion(image):
    """
    Get the pixel boundary regions for amps

    Args:
        image: desispec.image.Image object
    """
    from desispec.preproc import _parse_sec_keyword

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
    from desispec.preproc import _parse_sec_keyword

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


def countpix(image,nsig=None,ncounts=None):
    """
    Count the pixels above a given threshold.

    Threshold can be in n times sigma or counts.

    Args:
        image: 2d image array
        nsig: threshold in units of sigma, e.g 2 for 2 sigma
        ncounts: threshold in units of count, e.g 100
    """
    if nsig is not None:
        sig=np.std(image.ravel())
        counts_nsig=np.where(image.ravel() > nsig*sig)[0].shape[0]
        return counts_nsig
    if ncounts is not None:
        counts_thresh=np.where(image.ravel() > ncounts)[0].shape[0]
        return counts_thresh

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

def sky_resid(param, frame, skymodel, quick_look=False):
    """
    Algorithm for sky residual
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

    # Grab sky fibers on this frame
    skyfibers = np.where(frame.fibermap['OBJTYPE'] == 'SKY')[0]
    assert np.max(skyfibers) < 500  #- indices, not fiber numbers
    nfibers=len(skyfibers)
    qadict['NSKY_FIB'] = int(nfibers)

    #current_ivar=frame.ivar[skyfibers].copy()
    #flux = frame.flux[skyfibers]

    # Subtract
    #res = flux - skymodel.flux[skyfibers] # Residuals
    #res_ivar = util.combine_ivar(current_ivar, skymodel.ivar[skyfibers])

    #- Residuals
    res=frame.flux[skyfibers] #- as this frame is already sky subtracted
    res_ivar=frame.ivar[skyfibers]

    # Chi^2 and Probability
    chi2_fiber = np.sum(res_ivar*(res**2),1)
    chi2_prob = np.zeros(nfibers)
    for ii in range(nfibers):
        # Stats
        dof = np.sum(res_ivar[ii,:] > 0.)
        chi2_prob[ii] = scipy.stats.chisqprob(chi2_fiber[ii], dof)
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

    #- Residuals in wave and fiber axes
    qadict["MED_RESID_FIBER"]=np.median(res,axis=1)
    qadict["SKY_FIBERID"]=skyfibers.tolist()
    qadict["MED_RESID_WAVE"]=np.median(res,axis=0)

    #- Weighted average for each bin on all fibers
    qadict["WAVG_RES_WAVE"]= np.sum(res*res_ivar,0) / np.sum(res_ivar,0)

    #- Histograms for residual/sigma #- inherited from qa_plots.frame_skyres()
    binsz = param['BIN_SZ']
    gd_res = res_ivar > 0.
    devs = res[gd_res] * np.sqrt(res_ivar[gd_res])
    i0, i1 = int( np.min(devs) / binsz) - 1, int( np.max(devs) / binsz) + 1
    rng = tuple( binsz*np.array([i0,i1]) )
    nbin = i1-i0
    hist, edges = np.histogram(devs, range=rng, bins=nbin)

    qadict['DEVS_1D'] = hist.tolist() #- histograms for deviates
    qadict['DEVS_EDGES'] = edges.tolist() #- Bin edges

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
    """

    #- we calculate median and total S/N assuming no correlation bin by bin
    medsnr=np.zeros(flux.shape[0])
    #totsnr=np.zeros(flux.shape[0])
    for ii in range(flux.shape[0]):
        snr=flux[ii]*np.sqrt(ivar[ii])
        medsnr[ii]=np.median(snr)
        # totsnr[ii]=np.sqrt(np.sum(snr**2))
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

    qadict={"MEDIAN_SNR":medsnr,"MEDIAN_AMP_SNR":average_amp, "ELG_FIBERID":elgfibers.tolist(), "ELG_SNR_MAG": elg_snr_mag, "LRG_FIBERID":lrgfibers.tolist(), "LRG_SNR_MAG": lrg_snr_mag, "QSO_FIBERID": qsofibers.tolist(), "QSO_SNR_MAG": qso_snr_mag, "STAR_FIBERID": stdfibers.tolist(), "STAR_SNR_MAG":std_snr_mag}

    return qadict

def gauss(x,a,mu,sigma):
    """
    Gaussian fit of input data
    """
    return a*np.exp(-(x-mu)**2/(2*sigma**2))
