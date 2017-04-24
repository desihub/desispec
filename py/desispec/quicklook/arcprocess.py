import numpy as np
import scipy.optimize
from numpy.polynomial.legendre import Legendre, legval, legfit

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
        esigmas[jj]=(pcov[1,1]**0.5)*dw
 
    k=np.logical_and(~np.isnan(esigmas),esigmas!=np.inf)
    sigmas=sigmas[k]
    meanwaves=meanwaves[k]
    esigmas=esigmas[k]
    return meanwaves,emeanwaves,sigmas,esigmas

def fit_wsigmas(means,wsigmas,ewsigmas,npoly=5,domain=None):
    #- return callable legendre object
    wt=1/ewsigmas**2
    legfit = Legendre.fit(means, wsigmas, npoly, domain=None,w=wt)

    return legfit

def _gauss_pix(x,mean,sigma):
    x=(np.asarray(x,dtype=float)-mean)/(sigma*np.sqrt(2))
    dx=x[1]-x[0] #- uniform spacing
    edges= np.concatenate((x-dx/2, x[-1:]+dx/2))
    y=scipy.special.erf(edges)
    return (y[1:]-y[:-1])/2

def process_arc(frame,linelist=None,npoly=5,nbins=2):
    """
    frame: desispec.frame.Frame object, presumably resolution not evaluated. 
    linelist: line list to fit
    npoly: polynomial order for sigma expansion
    nbins: no of bins of the half of the fitting window
    return: desispec.frame.Frame object with resolution 
            desispec.psf.PSF object
    
    """
    nspec=frame.flux.shape[0]
    if linelist is None:
        linelist=[5854.1101,6404.018,7034.352,7440.9469] #- not final 

    coeffs=np.zeros(len(nspec),npoly+1) #- coeffs array
    for spec in range(len(nspec)):
        wave=frame.wavelength
        flux=frame.flux[spec]
        ivar=frame.ivar[spec]
        meanwaves,emeanwaves,sigmas,esigmas=sigmas_from_arc(wave,flux,ivar,linelist,n=nbins)
        domain=(np.min(wave),np.max(wave))
        thislegfit=fit_wsigmas(meanwaves,sigmas,esigmas,domain=domain,npoly=npoly)
        coeffs[spec]=thislegfit.coef
    
    return coeffs

def write_psffile(psfbootfile,wcoeffs,outfile):
    """ 
    extract psfbootfile, add wcoeffs, and make a new psf file
    """
    from astropy.io import fits
    psf=fits.open(psfbootfile)
    xcoeff=psf[0]
    ycoeff=psf[1]
    xsigma=psf[2]
    
    wsigma=fits.ImageHDU(wcoeffs,name='WSIGMA') #- in Angstrom
    hdulist=fits.HDUList([xcoeff,ycoeff,xsigma,wsigma])
    hdulist.writeto(outfile,clobber=True)
