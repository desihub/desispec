import numpy as np
import scipy.optimize
from numpy.polynomial.legendre import Legendre, legval, legfit
from desispec.quicklook import qlexceptions,qllogger
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

        spots=thisflux/thisflux.sum()
        errors=1./np.sqrt(thisivar)
        errors/=thisflux.sum()

        popt,pcov=scipy.optimize.curve_fit(_gauss_pix,thiswave,spots)
        meanwaves[jj]=popt[0]+linelist[jj]
        emeanwaves[jj]=pcov[0,0]**0.5
        sigmas[jj]=popt[1]
        esigmas[jj]=(pcov[1,1]**0.5)
 
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

    #- amend line list to only include lines in given wavelength range
    wave=frame.wave
    if wave[0] >= linelist[0]:
        noline_ind_lo=np.where(np.array(linelist)<=wave[0])
        linelist=linelist[np.max(noline_ind_lo[0])+1:len(linelist)-1]
        log.info("First {} line(s) outside wavelength range, skipping these".format(len(noline_ind_lo[0])))
    if wave[len(wave)-1] <= linelist[len(linelist)-1]:
        noline_ind_hi=np.where(np.array(linelist)>=wave[len(wave)-1])
        linelist=linelist[0:np.min(noline_ind_hi[0])-1]
        log.info("Last {} line(s) outside wavelength range, skipping these".format(len(noline_ind_hi[0])))

    for spec in range(nspec):
        flux=frame.flux[spec]
        ivar=frame.ivar[spec]
        meanwaves,emeanwaves,sigmas,esigmas=sigmas_from_arc(wave,flux,ivar,linelist,n=nbins)
        if domain is None:
            domain=(np.min(wave),np.max(wave))

        thislegfit=fit_wsigmas(meanwaves,sigmas,esigmas,domain=domain,npoly=npoly)
        coeffs[spec]=thislegfit.coef
    
    return coeffs

def write_psffile(psfbootfile,wcoeffs,outfile,wavestepsize=None):
    """ 
    extract psfbootfile, add wcoeffs, and make a new psf file preserving the traces etc. 
    psf module will load this 
    """
    from astropy.io import fits
    psf=fits.open(psfbootfile)
    xcoeff=psf[0]
    ycoeff=psf[1]
    xsigma=psf[2]
    
    wsigma=fits.ImageHDU(wcoeffs,name='WSIGMA')
    wsigma.header["PSFTYPE"]='boxcar'
    if wavestepsize is None:
        wavestepsize = 'NATIVE CCD GRID'
    wsigma.header["WAVESTEP"]=(wavestepsize,'Wavelength step size [Angstroms]')
    hdulist=fits.HDUList([xcoeff,ycoeff,xsigma,wsigma])
    hdulist.writeto(outfile,clobber=True)
