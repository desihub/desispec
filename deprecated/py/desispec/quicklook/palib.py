"""
desispec.quicklook.palib
========================

Low level functions to be from top level PAs.
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
    Pr=np.zeros((len(x2),len(x1)))

    e1 = np.zeros(len(x1)+1)
    e1[1:-1]=(x1[:-1]+x1[1:])/2.0  # calculate bin edges
    e1[0]=1.5*x1[0]-0.5*x1[1]
    e1[-1]=1.5*x1[-1]-0.5*x1[-2]
    e1lo = e1[:-1]  # make upper and lower bounds arrays vs. index
    e1hi = e1[1:]

    e2=np.zeros(len(x2)+1)
    e2[1:-1]=(x2[:-1]+x2[1:])/2.0  # bin edges for resampled grid
    e2[0]=1.5*x2[0]-0.5*x2[1]
    e2[-1]=1.5*x2[-1]-0.5*x2[-2]

    for ii in range(len(e2)-1): # columns
        #- Find indices in x1, containing the element in x2
        #- This is much faster than looping over rows

        k = np.where((e1lo<=e2[ii]) & (e1hi>e2[ii]))[0]
        # this where obtains single e1 edge just below start of e2 bin
        emin = e2[ii]
        emax = e1hi[k]
        if e2[ii+1] < emax : emax = e2[ii+1]
        dx = (emax-emin)/(e1hi[k]-e1lo[k])
        Pr[ii,k] = dx    # enter first e1 contribution to e2[ii]

        if e2[ii+1] > emax :
            # cross over to another e1 bin contributing to this e2 bin
            l = np.where((e1 < e2[ii+1]) & (e1 > e1hi[k]))[0]
            if len(l) > 0 :
               # several-to-one resample.  Just consider 3 bins max. case
               Pr[ii,k[0]+1] = 1.0  # middle bin fully contained in e2
               q = k[0]+2
            else : q = k[0]+1  # point to bin partially contained in current e2 bin

            try:
                emin = e1lo[q]
                emax = e2[ii+1]
                dx = (emax-emin)/(e1hi[q]-e1lo[q])
                Pr[ii,q] = dx
            except:
                pass

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

    Pr=project(wave,outwave)
    n=len(wave)
    newflux=Pr.dot(flux)
    #- convert back to df/dx (per angstrom) sampled at outwave
    newflux/=np.gradient(outwave) #- per angstrom
    if ivar is None:
        return newflux
    else:
        ivar = ivar/(np.gradient(wave))**2.0
        newvar=Pr.dot(ivar**(-1.0)) #- maintaining Total S/N
        # RK:  this is just a kludge until we more robustly ensure newvar is correct
        k = np.where(newvar <= 0.0)[0]
        newvar[k] = 0.0000001  # flag bins with no contribution from input grid
        newivar=1/newvar
        # newivar[k] = 0.0

        #- convert to per angstrom
        newivar*=(np.gradient(outwave))**2.0
        return newflux, newivar

def get_resolution(wave,nspec,tset,usesigma=False):
    """
    Calculates approximate resolution values at given wavelengths in the format that can directly
    feed resolution data of desispec.frame.Frame object.

    wave: wavelength array
    nsepc: no of spectra (int)
    tset: desispec.xytraceset like object
    usesigma: allows to use sigma from psf file for resolution computation.

    returns : resolution data (nspec,nband,nwave); nband = 1 for usesigma = False, otherwise nband=21
    """
    #from desispec.resolution import Resolution
    from desispec.quicklook.qlresolution import QuickResolution
    nwave=len(wave)
    if usesigma:
        nband=21
    else:
        nband=1 # only for dimensionality purpose of data model.
    resolution_data=np.zeros((nspec,nband,nwave))

    if usesigma: #- use sigmas for resolution based on psffile type
        for ispec in range(nspec):
            thissigma=tset.ysig_vs_wave(ispec,wave) #- in pixel units
            Rsig=QuickResolution(sigma=thissigma,ndiag=nband)
            resolution_data[ispec]=Rsig.data

    return resolution_data

def apply_flux_calibration(frame,fluxcalib):
    """
    Apply flux calibration to sky subtracted qframe
    Use offline algorithm, but assume qframe object is input
    and that it is on native ccd wavelength grid
    Calibration vector is resampled to frame wavelength grid

    frame: QFrame object
    fluxcalib: FluxCalib object

    Modifies frame.flux and frame.ivar
    """
    from desispec.quicklook.palib import resample_spec

    nfibers=frame.nspec

    resample_calib=[]
    resample_ivar=[]
    for i in range(nfibers):
        rescalib,resivar=resample_spec(fluxcalib.wave,fluxcalib.calib[i],frame.wave[i],ivar=fluxcalib.ivar[i])
        resample_calib.append(rescalib)
        resample_ivar.append(resivar)
    fluxcalib.calib=np.array(resample_calib)
    fluxcalib.ivar=np.array(resample_ivar)

    C = fluxcalib.calib
    frame.flux=frame.flux*(C>0)/(C+(C==0))
    frame.ivar*=(fluxcalib.ivar>0)*(C>0)
    for j in range(nfibers):
        ok=np.where(frame.ivar[j]>0)[0]
        if ok.size>0:
            frame.ivar[j,ok]=1./(1./(frame.ivar[j,ok]*C[j,ok]**2)+frame.flux[j,ok]**2/(fluxcalib.ivar[j,ok]*C[j,ok]**4))

