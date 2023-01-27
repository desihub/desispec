"""
desispec.quicklook.arcprocess
=============================

"""
import numpy as np
import scipy.optimize
from numpy.polynomial.legendre import Legendre, legval, legfit
from desispec.quicklook import qlexceptions,qllogger
from desispec.io import read_xytraceset, write_xytraceset
from specter.util.traceset import TraceSet,fit_traces

qlog=qllogger.QLLogger("QuickLook",20)
log=qlog.getlog()

def sigmas_from_arc(wave,flux,ivar,linelist,n=2):
    """
    Gaussian fitting of listed arc lines and return corresponding sigmas in pixel units
    Args:
    linelist: list of lines (A) for which fit is to be done
    n: fit region half width (in bin units): n=2 bins => (2*n+1)=5 bins fitting window.
    """

    nwave=wave.shape

    #- select the closest match to given lines
    ind=[(np.abs(wave-line)).argmin() for line in linelist]

    #- fit gaussian obout the peaks
    meanwaves=np.zeros(len(ind))
    emeanwaves=np.zeros(len(ind))
    sigmas=np.zeros(len(ind))
    esigmas=np.zeros(len(ind))

    for jj,index in enumerate(ind):
        thiswave=wave[index-n:index+n+1]-linelist[jj] #- fit window about 0
        thisflux=flux[index-n:index+n+1]
        thisivar=ivar[index-n:index+n+1]

        #RS: skip lines with zero flux
        if 0. not in thisflux:
            spots=thisflux/thisflux.sum()
            try:
                popt,pcov=scipy.optimize.curve_fit(_gauss_pix,thiswave,spots)
                meanwaves[jj]=popt[0]+linelist[jj]
                if pcov[0,0] >= 0.:
                    emeanwaves[jj]=pcov[0,0]**0.5
                sigmas[jj]=popt[1]
                if pcov[1,1] >= 0.:
                    esigmas[jj]=(pcov[1,1]**0.5)
            except:
                pass

    k=np.logical_and(~np.isnan(esigmas),esigmas!=np.inf)
    sigmas=sigmas[k]
    meanwaves=meanwaves[k]
    esigmas=esigmas[k]
    return meanwaves,emeanwaves,sigmas,esigmas

def fit_wsigmas(means,wsigmas,ewsigmas,npoly=2,domain=None):
    #- return callable legendre object
    wt=1/ewsigmas**2
    legfit = Legendre.fit(means, wsigmas, npoly, domain=domain,w=wt)

    return legfit

def _gauss_pix(x,mean,sigma):
    x=(np.asarray(x,dtype=float)-mean)/(sigma*np.sqrt(2))
    dx=x[1]-x[0] #- uniform spacing
    edges= np.concatenate((x-dx/2, x[-1:]+dx/2))
    y=scipy.special.erf(edges)
    return (y[1:]-y[:-1])/2

def process_arc(frame,linelist=None,npoly=2,nbins=2,domain=None):
    """
    frame: desispec.frame.Frame object, preumably resolution not evaluated.
    linelist: line list to fit
    npoly: polynomial order for sigma expansion
    nbins: no of bins for the half of the fitting window
    return: coefficients of the polynomial expansion

    """

    if domain is None :
        raise ValueError("domain must be given in process_arc")

    nspec=frame.flux.shape[0]
    if linelist is None:
        camera=frame.meta["CAMERA"]
        #- load arc lines
        from desispec.bootcalib import load_arcline_list, load_gdarc_lines,find_arc_lines
        llist=load_arcline_list(camera)
        dlamb,gd_lines=load_gdarc_lines(camera,llist)
        linelist=gd_lines
        #linelist=[5854.1101,6404.018,7034.352,7440.9469] #- not final
        log.info("No line list configured. Fitting for lines {}".format(linelist))
    coeffs=np.zeros((nspec,npoly+1)) #- coeffs array

    for spec in range(nspec):
        #- Allow arc processing to use either QL or QP extraction
        if isinstance(frame.wave[0],float):
            wave=frame.wave
        else:
            wave=frame.wave[spec]

        flux=frame.flux[spec]
        ivar=frame.ivar[spec]

        #- amend line list to only include lines in given wavelength range
        if wave[0] >= linelist[0]:
            noline_ind_lo=np.where(np.array(linelist)<=wave[0])
            linelist=linelist[np.max(noline_ind_lo[0])+1:len(linelist)-1]
            log.info("First {} line(s) outside wavelength range, skipping these".format(len(noline_ind_lo[0])))
        if wave[len(wave)-1] <= linelist[len(linelist)-1]:
            noline_ind_hi=np.where(np.array(linelist)>=wave[len(wave)-1])
            linelist=linelist[0:np.min(noline_ind_hi[0])-1]
            log.info("Last {} line(s) outside wavelength range, skipping these".format(len(noline_ind_hi[0])))

        meanwaves,emeanwaves,sigmas,esigmas=sigmas_from_arc(wave,flux,ivar,linelist,n=nbins)
        if domain is None:
            domain=(np.min(wave),np.max(wave))

        # RS: if Gaussian couldn't be fit to a line, don't do legendre fit for fiber
        if 0. in sigmas or 0. in esigmas:
            pass
        else:
            try:
                thislegfit=fit_wsigmas(meanwaves,sigmas,esigmas,domain=domain,npoly=npoly)
                coeffs[spec]=thislegfit.coef
            except:
                pass

    # need to return the wavemin and wavemax of the fit
    return coeffs,domain[0],domain[1]

def write_psffile(infile,wcoeffs,wcoeffs_wavemin,wcoeffs_wavemax,outfile,wavestepsize=None):
    """
    extract psf file, add wcoeffs, and make a new psf file preserving the traces etc.
    psf module will load this
    """

    tset = read_xytraceset(infile)

    # convert wsigma to ysig ...
    nfiber    = wcoeffs.shape[0]
    ncoef     = wcoeffs.shape[1]
    nw        = 100 # need a larger number than ncoef to get an accurate dydw from the gradients

    # wcoeffs and tset do not necessarily have the same wavelength range
    wave      = np.linspace(tset.wavemin,tset.wavemax,nw)
    wsig_set  = TraceSet(wcoeffs,[wcoeffs_wavemin,wcoeffs_wavemax])
    wsig_vals = np.zeros((nfiber,nw))
    for f in range(nfiber) :
        y_vals = tset.y_vs_wave(f,wave)
        dydw   = np.gradient(y_vals)/np.gradient(wave)
        wsig_vals[f]=wsig_set.eval(f,wave)*dydw
    tset.ysig_vs_wave_traceset = fit_traces(wave, wsig_vals, deg=ncoef-1, domain=(tset.wavemin,tset.wavemax))

    write_xytraceset(outfile,tset)

