"""
desispec.quicklook.palib
Low level functions to be from top level PAs
"""
import numpy as np
from desispec.quicklook import qlexceptions,qllogger
qlog=qllogger.QLLogger("QuickLook",20)
log=qlog.getlog()


def project(x1,x2):
    """
    return a projection matrix so that arrays are related by linear interpolation
    x1: Array with one binning
    x2: new binning
    
    Return Pr: x1= Pr.dot(x2) in the overlap region
    """
    x1=np.sort(x1)
    x2=np.sort(x2)
    Pr=np.zeros((len(x1),len(x2)))
    for ii in range(len(x2)-1): # columns
        #- Find indices in x1, containing the element in x2 
        #- This is much faster than looping over rows
        k=np.where((x1>=x2[ii]) & (x1<=x2[ii+1]))[0] 
        if len(k)>0:
            dx=(x1[k]-x2[ii])/(x2[ii+1]-x2[ii])
            Pr[k,ii]=1-dx
            Pr[k,ii+1]=dx
    #- edge: 
    if x2[-1]==x1[-1]:
        Pr[-1,-1]=1
    return Pr

def resample_spec(wave,flux,outwave,ivar=None):
    """
    rebinning conserving S/N
    Algorithm is based on http://www.ast.cam.ac.uk/%7Erfc/vpfit10.2.pdf
    Appendix: B.1

    Args: 
    wave : original wavelength array (expected (but not limited) to be native CCD pixel wavelength grid
    outwave: new wavelength array: expected (but not limited) to be uniform binning 
    flux : df/dx (Flux per A) sampled at x
    ivar : ivar in original binning. If not None, ivar in new binning is returned. 

    Note: 
    Full resolution computation for resampling is expensive for quicklook.

    desispec.interpolation.resample_flux using weights by ivar does not conserve total S/N. 
    Tests with arc lines show much narrow spectral profile, thus not giving realistic psf resolutions
    This algorithm gives the same resolution as obtained for native CCD binning, i.e, resampling has 
    insignificant effect. Details,plots in the arc processing note.
    """
    #- convert flux to per bin before projecting to new bins
    flux=flux*np.gradient(wave) 
    ivar=ivar/(np.gradient(wave))**2

    Pr=project(wave,outwave)
    n=len(wave)
    
    newflux=Pr.T.dot(flux)
    #- convert back to df/dx (per angstrom) sampled at outwave
    newflux/=np.gradient(outwave) #- per angstrom
    if ivar is None:
        return newflux
    else:
        newvar=Pr.T.dot(ivar**(-1.)) #- maintaining Total S/N
        newivar=1/newvar

        #- convert to per angstrom
        newivar*=(np.gradient(outwave))**2
        return newflux, newivar


def get_resolution(wave,nspec,psf,usesigma=False):
    """
    Calculates approximate resolution values at given wavelengths in the format that can directly
    feed resolution data of desispec.frame.Frame object. 

    wave: wavelength array
    nsepc: no of spectra (int)
    psf: desispec.psf.PSF like object
    usesigma: allows to use sigma from psf file for resolution computation. 
    If psf file is psfboot, uses per fiber xsigma. 
    If psf file is from QL arcs processing, uses wsigma 

    returns : resolution data (nspec,nband,nwave); nband = 1 for usesigma = False, otherwise nband=21
    """
    from desispec.resolution import Resolution

    nwave=len(wave)
    if usesigma:
        nband=21
    else:
        nband=1 # only for dimensionality purpose of data model.
    resolution_data=np.zeros((nspec,nband,nwave))

    if usesigma: #- use sigmas for resolution based on psffile type

        if hasattr(psf,'wcoeff'): #- use if have wsigmas
            log.info("Getting resolution from wsigmas from arc lines PSF")
            for ispec in range(nspec):
                thissigma=psf.wdisp(ispec,wave)/psf.angstroms_per_pixel(ispec,wave) #- in pixel units
                Rsig=Resolution(thissigma)
                resolution_data[ispec]=Rsig.data
        else:

            if hasattr(psf,'xsigma_boot'): #- only use if xsigma comes from psfboot
                log.info("Getting resolution matrix band diagonal elements from constant Gaussian Xsigma")
                for ispec in range(nspec):
                    thissigma=psf.xsigma(ispec,wave) 
                    Rsig=Resolution(thissigma)
                    resolution_data[ispec]=Rsig.data

    return resolution_data

