"""
desispec.qproc.qarc
===================

"""
import numpy as np
import scipy.optimize

from numpy.polynomial.legendre import Legendre, legval, legfit
from specter.util.traceset import TraceSet,fit_traces
from desiutil.log import get_logger

# largely inspired from quicklook.arcprocess.py but duplicated here to use qframe

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

def _gauss_pix(x,mean,sigma):
    x=(np.asarray(x,dtype=float)-mean)/(sigma*np.sqrt(2))
    dx=x[1]-x[0] #- uniform spacing
    edges= np.concatenate((x-dx/2, x[-1:]+dx/2))
    y=scipy.special.erf(edges)
    return (y[1:]-y[:-1])/2

def process_arc(qframe,xytraceset,linelist=None,npoly=2,nbins=2):

    """
    qframe: desispec.qframe.QFrame object
    xytraceset : desispec.xytraceset.XYTraceSet object
    linelist: line list to fit
    npoly: polynomial order for sigma expansion
    nbins: no of bins for the half of the fitting window
    return: xytraceset (with ysig vs wave)
    """

    log = get_logger()

    if linelist is None:

        if qframe.meta is None or "CAMERA" not in qframe.meta :
            log.error("no information about camera in qframe so I don't know which lines to use")
            raise RuntimeError("no information about camera in qframe so I don't know which lines to use")

        camera=qframe.meta["CAMERA"]
        #- load arc lines
        from desispec.bootcalib import load_arcline_list, load_gdarc_lines,find_arc_lines
        llist=load_arcline_list(camera)
        dlamb,gd_lines=load_gdarc_lines(camera,llist)
        linelist=gd_lines
        log.info("No line list configured. Fitting for lines {}".format(linelist))

    tset=xytraceset

    assert(qframe.nspec == tset.nspec)

    tset.ysig_vs_wave_traceset = TraceSet(np.zeros((tset.nspec,npoly+1)),[tset.wavemin,tset.wavemax])

    for spec in range(tset.nspec):
        spec_wave     = qframe.wave[spec]
        spec_linelist = linelist[(linelist>spec_wave[0])&(linelist<spec_wave[-1])]
        meanwaves,emeanwaves,sigmas,esigmas=sigmas_from_arc(spec_wave,qframe.flux[spec],qframe.ivar[spec],spec_linelist,n=nbins)

        # convert from wavelength A unit to CCD pixel for consistency with specex PSF
        y = tset.y_vs_wave(spec,spec_wave)
        dydw = np.interp(meanwaves,spec_wave,np.gradient(y)/np.gradient(spec_wave))
        sigmas *= dydw # A -> pixels
        esigmas *= dydw # A -> pixels

        ok=(sigmas>0)&(esigmas>0)

        try:
            thislegfit = Legendre.fit(meanwaves[ok], sigmas[ok], npoly, domain=[tset.wavemin,tset.wavemax],w=1./esigmas[ok]**2)
            tset.ysig_vs_wave_traceset._coeff[spec] = thislegfit.coef
        except:
            log.error("legfit of psf width failed for spec {}".format(spec))

        wave=np.linspace(tset.wavemin,tset.wavemax,20)
        #plt.plot(wave,tset.ysig_vs_wave(spec,wave))

    #plt.show()
    return xytraceset
