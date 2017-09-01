
from __future__ import absolute_import, division


import sys
import argparse
import numpy as np
from numpy.linalg.linalg import LinAlgError
import astropy.io.fits as pyfits
from numpy.polynomial.legendre import legval,legfit
from scipy.signal import fftconvolve

import specter.psf
from desispec.io import read_image
from desiutil.log import get_logger
from desispec.linalg import cholesky_solve,cholesky_solve_and_invert
from desispec.interpolation import resample_flux

def read_psf_and_traces(psf_filename) :
    """
    Reads PSF and traces in PSF fits file
    
    Args:
        psf_filename : Path to input fits file which has to contain XTRACE and YTRACE HDUs
    Returns:
        psf : specter PSF object
        xtrace : 2D np.array of shape (nfibers,ncoef) containing Legendre coefficents for each fiber to convert wavelenght to XCCD
        ytrace : 2D np.array of shape (nfibers,ncoef) containing Legendre coefficents for each fiber to convert wavelenght to YCCD
        wavemin : float
        wavemax : float. wavemin and wavemax are used to define a reduced variable legx(wave,wavemin,wavemax)=2*(wave-wavemin)/(wavemax-wavemin)-1
                  used to compute the traces, xccd=legval(legx(wave,wavemin,wavemax),xtrace[fiber])
    
    """

    log=get_logger()

    psf=None
    xtrace=None
    ytrace=None
    wavemin=None
    wavemax=None
    wavemin2=None
    wavemax2=None

    fits_file = pyfits.open(psf_filename)
    
    try :
        psftype=fits_file[0].header["PSFTYPE"]
    except KeyError :
        psftype=""
    if psftype=="GAUSS-HERMITE" :
        psf = specter.psf.GaussHermitePSF(psf_filename)
    elif psftype=="SPOTGRID" :
        psf = specter.psf.SpotGridPSF(psf_filename)

    # now read trace coefficients
    if "XTRACE" in fits_file :
        xtrace=fits_file["XTRACE"].data
        ytrace=fits_file["YTRACE"].data
        wavemin=fits_file["XTRACE"].header["WAVEMIN"]
        wavemax=fits_file["XTRACE"].header["WAVEMAX"]
        wavemin2=fits_file["YTRACE"].header["WAVEMIN"]
        wavemax2=fits_file["YTRACE"].header["WAVEMAX"]
        
    else :
        log.info("psf is a '%s'"%psftype)
        if psftype == "bootcalib" :    
            wavemin = fits_file[0].header["WAVEMIN"]
            wavemax = fits_file[0].header["WAVEMAX"]
            xcoef   = fits_file[0].data
            ycoef   = fits_file[1].data
            xsig    = fits_file[2].data
        elif psftype == "GAUSS-HERMITE" :
            table=fits_file[1].data        
            i=np.where(table["PARAM"]=="X")[0][0]
            wavemin=table["WAVEMIN"][i]
            wavemax=table["WAVEMAX"][i]
            xtrace=table["COEFF"][i]
            i=np.where(table["PARAM"]=="Y")[0][0]
            ytrace=table["COEFF"][i]
            wavemin2=table["WAVEMIN"][i]
            wavemax2=table["WAVEMAX"][i]
            
    if xtrace is None or ytrace is None :
        raise ValueError("could not find XTRACE and YTRACE in psf file %s"%psf_filename)
    if wavemin != wavemin2 :
        raise ValueError("XTRACE and YTRACE don't have same WAVEMIN %f %f"%(wavemin,wavemin2))
    if wavemax != wavemax2 :
        raise ValueError("XTRACE and YTRACE don't have same WAVEMAX %f %f"%(wavemax,wavemax2))
    if xtrace.shape[0] != ytrace.shape[0] :
        raise ValueError("XTRACE and YTRACE don't have same number of fibers %d %d"%(xtrace.shape[0],ytrace.shape[0]))
    
    fits_file.close()
    
    return psf,xtrace,ytrace,wavemin,wavemax

   
    
    
def write_traces_in_psf(input_psf_filename,output_psf_filename,xcoef,ycoef,wavemin,wavemax) :
    """
    Writes traces in a PSF.
    
    Args:
        input_psf_filename : Path to input fits file which has to contain XTRACE and YTRACE HDUs
        output_psf_filename : Path to output fits file which has to contain XTRACE and YTRACE HDUs
        xcoef : 2D np.array of shape (nfibers,ncoef) containing Legendre coefficents for each fiber to convert wavelenght to XCCD
        ycoef : 2D np.array of shape (nfibers,ncoef) containing Legendre coefficents for each fiber to convert wavelenght to YCCD
        wavemin : float
        wavemax : float. wavemin and wavemax are used to define a reduced variable legx(wave,wavemin,wavemax)=2*(wave-wavemin)/(wavemax-wavemin)-1
                  used to compute the traces, xccd=legval(legx(wave,wavemin,wavemax),xtrace[fiber])    
    """
    psf_fits=pyfits.open(input_psf_filename)

    psftype=psf_fits[0].header["PSFTYPE"]
    if psftype=="GAUSS-HERMITE" :             
        i=np.where(psf_fits[1].data["PARAM"]=="X")[0][0]
        psf_fits[1].data["COEFF"][i][:xcoef.shape[0]]=xcoef
        i=np.where(psf_fits[1].data["PARAM"]=="Y")[0][0]
        psf_fits[1].data["COEFF"][i][:ycoef.shape[0]]=ycoef
    
    if "XTRACE" in psf_fits :
        psf_fits["XTRACE"].data = xcoef
        psf_fits["XTRACE"].header["WAVEMIN"] = wavemin
        psf_fits["XTRACE"].header["WAVEMAX"] = wavemax
    if "YTRACE" in psf_fits :
        psf_fits["YTRACE"].data = ycoef
        psf_fits["YTRACE"].header["WAVEMIN"] = wavemin
        psf_fits["YTRACE"].header["WAVEMAX"] = wavemax
    
    psf_fits.writeto(output_psf_filename,clobber=True)
    
    
def legx(wave,wavemin,wavemax) :
    """ 
    Reduced coordinate (range [-1,1]) for calls to legval and legfit

    Args:
        wave : ND np.array
        wavemin : float, min. val
        wavemax : float, max. val
    Returns:
        array of same shape as wave
    """
    
    return 2.*(wave-wavemin)/(wavemax-wavemin)-1.

# beginning of routines for cross-correlation method for trace shifts  

def boxcar_extraction(xcoef,ycoef,wavemin,wavemax, image, fibers=None, width=7) :    
    """
    Fast boxcar extraction of spectra from a preprocessed image and a trace set
    
    Args:
        xcoef : 2D np.array of shape (nfibers,ncoef) containing Legendre coefficents for each fiber to convert wavelenght to XCCD
        ycoef : 2D np.array of shape (nfibers,ncoef) containing Legendre coefficents for each fiber to convert wavelenght to YCCD
        wavemin : float
        wavemax : float. wavemin and wavemax are used to define a reduced variable legx(wave,wavemin,wavemax)=2*(wave-wavemin)/(wavemax-wavemin)-1
                  used to compute the traces, xccd=legval(legx(wave,wavemin,wavemax),xtrace[fiber]) 
        image : DESI preprocessed image object

    Optional:   
        fibers : 1D np.array of int (default is all fibers, the first fiber is always = 0)
        width  : extraction boxcar width, default is 7

    Returns:
        flux :  2D np.array of shape (nfibers,n0=image.shape[0]), sum of pixel values per row of length=width per fiber
        ivar :  2D np.array of shape (nfibers,n0), ivar[f,j] = 1/( sum_[j,b:e] (1/image.ivar) ), ivar=0 if at least 1 pixel in the row has image.ivar=0 or image.mask!=0
        wave :  2D np.array of shape (nfibers,n0), determined from the traces
    """
    log=get_logger()
    log.info("Starting boxcar extraction...")
    
    if fibers is None :
        fibers = np.arange(psf.nspec)
    
    log.info("wavelength range : [%f,%f]"%(wavemin,wavemax))
    
    if image.mask is not None :
        image.ivar *= (image.mask==0)
    
    #  Applying a mask that keeps positive value to get the Variance by inversing the inverse variance.
    var=np.zeros(image.ivar.size)
    ok=image.ivar.ravel()>0
    var[ok] = 1./image.ivar.ravel()[ok]
    var=var.reshape(image.ivar.shape)

    badimage=(image.ivar==0)
    
    n0 = image.pix.shape[0]
    n1 = image.pix.shape[1]
    
    frame_flux = np.zeros((fibers.size,n0))
    frame_ivar = np.zeros((fibers.size,n0))
    frame_wave = np.zeros((fibers.size,n0))
    xx         = np.tile(np.arange(n1),(n0,1))
    hw = width//2
    
    ncoef=ycoef.shape[1]
    twave=np.linspace(wavemin, wavemax, ncoef+2)
    
    for f,fiber in enumerate(fibers) :
        log.info("extracting fiber #%03d"%fiber)
        y_of_wave     = legval(legx(twave, wavemin, wavemax), ycoef[fiber])
        coef          = legfit(legx(y_of_wave, 0, n0), twave, deg=ncoef) # add one deg
        frame_wave[f] = legval(legx(np.arange(n0).astype(float), 0, n0), coef)
        x_of_y        = np.floor( legval(legx(frame_wave[f], wavemin, wavemax), xcoef[fiber]) + 0.5 ).astype(int)
        mask=((xx.T>=x_of_y-hw)&(xx.T<=x_of_y+hw)).T
        frame_flux[f]=image.pix[mask].reshape((n0,width)).sum(-1)
        tvar=var[mask].reshape((n0,width)).sum(-1)
        frame_ivar[f]=(tvar>0)/(tvar+(tvar==0))
        bad=(badimage[mask].reshape((n0,width)).sum(-1))>0
        frame_ivar[f,bad]=0.
    
    return frame_flux, frame_ivar, frame_wave

def resample_boxcar_frame(frame_flux,frame_ivar,frame_wave,oversampling=2) :
    """
    Resamples the spectra in a frame obtained with boxcar extraction to the same wavelength grid, with oversampling.
    Uses resample_flux routine.
    
    Args:
        frame_flux :  2D np.array of shape (nfibers,nwave), sum of pixel values per row of length=width per fiber
        frame_ivar :  2D np.array of shape (nfibers,nwave), ivar[f,j] = 1/( sum_[j,b:e] (1/image.ivar) ), ivar=0 if at least 1 pixel in the row has image.ivar=0 or image.mask!=0
        frame_wave :  2D np.array of shape (nfibers,nwave), determined from the traces
    Optional:   
        oversampling : int , oversampling factor , default is 2
    
    Returns:
        flux :  2D np.array of shape (nfibers,nwave*oversampling) 
        ivar :  2D np.array of shape (nfibers,nwave*oversampling)
        frame_wave :  1D np.array of size (nwave*oversampling)    
    """
    log=get_logger()
    
    log.info("resampling with oversampling")
    nfibers=frame_flux.shape[0]
    wave=frame_wave[nfibers//2]
    dwave=np.median(np.gradient(frame_wave))/oversampling
    wave=np.linspace(wave[0],wave[-1],int((wave[-1]-wave[0])/dwave))
    nwave=wave.size
    
    flux=np.zeros((nfibers,nwave))
    ivar=np.zeros((nfibers,nwave))
    for i in range(nfibers) :
        log.info("resampling fiber #%03d"%i)
        flux[i],ivar[i] = resample_flux(wave, frame_wave[i],frame_flux[i],frame_ivar[i])
    return flux,ivar,wave




def compute_dy_from_spectral_cross_correlation(flux,wave,refflux,ivar=None,hw=3.,deg=2) :
    """
    Measure y offsets from two spectra expected to be on the same wavelength grid.
    refflux is the assumed well calibrated spectrum.
    A relative flux calibration of the two spectra is done internally.
    
    Args:
        flux    : 1D array of spectral flux as a function of wavelenght
        wave    : 1D array of wavelength (in Angstrom)
        refflux : 1D array of reference spectral flux
    Optional:   
        ivar   : 1D array of inverse variance of flux
        hw     : half width in Angstrom of the cross-correlation chi2 scan, default=3A corresponding approximatly to 5 pixels for DESI 
        deg    : degree of polynomial fit as a function of wavelength, only used to find and mask outliers
        
    Returns:
        x  : 1D array of x coordinates on CCD (axis=1 in numpy image array, AXIS=0 in FITS, cross-dispersion axis = fiber number direction) 
        y  : 1D array of y coordinates on CCD (axis=0 in numpy image array, AXIS=1 in FITS, wavelength dispersion axis)        
        dx : 1D array of shifts along x coordinates on CCD
        ex : 1D array of uncertainties on dx 
        fiber : 1D array of fiber ID (first fiber = 0) 
        wave  : 1D array of wavelength
    """
    
    # absorb differences of calibration (fiberflat not yet applied)
    x=(wave-wave[wave.size//2])/50.
    kernel=np.exp(-x**2/2)
    f1=fftconvolve(flux,kernel,mode='same')
    f2=fftconvolve(refflux,kernel,mode='same')
    scale=f1/f2
    refflux *= scale
    
    error_floor=0.001 #A
    
    if ivar is None :
        ivar=np.ones(flux.shape)
    dwave=wave[1]-wave[0]
    ihw=int(hw/dwave)+1
    chi2=np.zeros((2*ihw+1))
    for i in range(2*ihw+1) :
        d=i-ihw
        b=ihw+d
        e=-ihw+d
        if e==0 :
            e=wave.size
        chi2[i] = np.sum(ivar[ihw:-ihw]*(flux[ihw:-ihw]-refflux[b:e])**2)

    #import matplotlib.pyplot as plt
    #plt.figure("%d"%int(np.mean(wave)))
    #plt.plot(dwave*(np.arange(-ihw,ihw+1)),chi2)
    
    i=np.argmin(chi2)
    b=i-1
    e=i+2
    if b<0 : 
        b=0
        e=b+3
    if e>2*ihw+1 :
        e=2*ihw+1
        b=e-3
    x=dwave*(np.arange(b,e)-ihw)
    c=np.polyfit(x,chi2[b:e],deg)
    if c[0]>0 :
        delta=-c[1]/(2.*c[0])
        sigma=np.sqrt(1./c[0] + error_floor**2)
    else :
        delta=0.
        sigma=100.
    return delta,sigma


def compute_dy_from_spectral_cross_correlations_of_frame(flux, ivar, wave , xcoef, ycoef, wavemin, wavemax, reference_flux , n_wavelength_bins = 4) :
    """
    Measures y offsets from a set of resampled spectra and a reference spectrum that are on the same wavelength grid.
    reference_flux is the assumed well calibrated spectrum.
    Calls compute_dy_from_spectral_cross_correlation per fiber
    
    Args:
        flux    : 2D np.array of shape (nfibers,nwave)
        ivar    : 2D np.array of shape (nfibers,nwave) , inverse variance of flux
        wave    : 1D array of wavelength (in Angstrom) of size nwave    
        refflux : 1D array of reference spectral flux of size nwave
    Optional:   
        n_wavelength_bins : number of bins along wavelength  
        
    Returns:
        x  : 1D array of x coordinates on CCD (axis=1 in numpy image array, AXIS=0 in FITS, cross-dispersion axis = fiber number direction) 
        y  : 1D array of y coordinates on CCD (axis=0 in numpy image array, AXIS=1 in FITS, wavelength dispersion axis)        
        dy : 1D array of shifts along y coordinates on CCD
        ey : 1D array of uncertainties on dy 
        fiber : 1D array of fiber ID (first fiber = 0) 
        wave  : 1D array of wavelength
    
    """
    log=get_logger()

    x_for_dy=np.array([])
    y_for_dy=np.array([])
    dy=np.array([])
    ey=np.array([])
    fiber_for_dy=np.array([])
    wave_for_dy=np.array([])
    
    nfibers = flux.shape[0]
    
    for fiber in range(nfibers) :
        log.info("computing dy for fiber #%03d"%fiber)
        
        for b in range(n_wavelength_bins) :
            wmin=wave[0]+((wave[-1]-wave[0])/n_wavelength_bins)*b
            if b<n_wavelength_bins-1 :
                wmax=wave[0]+((wave[-1]-wave[0])/n_wavelength_bins)*(b+1)
            else :
                wmax=wave[-1]
            ok=(wave>=wmin)&(wave<=wmax)
            sw=np.sum(ivar[fiber,ok]*flux[fiber,ok]*(flux[fiber,ok]>0))
            if sw<=0 :
                continue
            dwave,err = compute_dy_from_spectral_cross_correlation(flux[fiber,ok],wave[ok],reference_flux[ok],ivar=ivar[fiber,ok],hw=3.)
            block_wave = np.sum(ivar[fiber,ok]*flux[fiber,ok]*(flux[fiber,ok]>0)*wave[ok])/sw
            if err > 1 :
                continue
            rw = legx(block_wave,wavemin,wavemax)
            tx = legval(rw,xcoef[fiber])
            ty = legval(rw,ycoef[fiber])
            eps=0.1
            yp = legval(legx(block_wave+eps,wavemin,wavemax),ycoef[fiber])
            dydw = (yp-ty)/eps
            tdy = -dwave*dydw
            tey = err*dydw
            
            x_for_dy=np.append(x_for_dy,tx)
            y_for_dy=np.append(y_for_dy,ty)
            dy=np.append(dy,tdy)
            ey=np.append(ey,tey)
            fiber_for_dy=np.append(fiber_for_dy,fiber)
            wave_for_dy=np.append(wave_for_dy,block_wave)

    return x_for_dy,y_for_dy,dy,ey,fiber_for_dy,wave_for_dy

def compute_dy_using_boxcar_extraction(xcoef,ycoef,wavemin,wavemax, image, fibers, width=7, degyy=2) :
    """
    Measures y offsets (internal wavelength calibration) from a preprocessed image and a trace set using a cross-correlation of boxcar extracted spectra.
    Uses boxcar_extraction , resample_boxcar_frame , compute_dy_from_spectral_cross_correlations_of_frame
    
    Args:
        xcoef : 2D np.array of shape (nfibers,ncoef) containing Legendre coefficents for each fiber to convert wavelenght to XCCD
        ycoef : 2D np.array of shape (nfibers,ncoef) containing Legendre coefficents for each fiber to convert wavelenght to YCCD
        wavemin : float
        wavemax : float. wavemin and wavemax are used to define a reduced variable legx(wave,wavemin,wavemax)=2*(wave-wavemin)/(wavemax-wavemin)-1
                  used to compute the traces, xccd=legval(legx(wave,wavemin,wavemax),xtrace[fiber]) 
        image : DESI preprocessed image object

    Optional:   
        fibers : 1D np.array of int (default is all fibers, the first fiber is always = 0)
        width  : int, extraction boxcar width, default is 7
        degyy  : int, degree of polynomial fit of shifts as a function of y, used to reject outliers.
    
    Returns:
        x  : 1D array of x coordinates on CCD (axis=1 in numpy image array, AXIS=0 in FITS, cross-dispersion axis = fiber number direction) 
        y  : 1D array of y coordinates on CCD (axis=0 in numpy image array, AXIS=1 in FITS, wavelength dispersion axis)        
        dy : 1D array of shifts along y coordinates on CCD
        ey : 1D array of uncertainties on dy 
        fiber : 1D array of fiber ID (first fiber = 0) 
        wave  : 1D array of wavelength

    """

    log=get_logger()
    
    # boxcar extraction
    boxcar_flux, boxcar_ivar, boxcar_wave = boxcar_extraction(xcoef,ycoef,wavemin,wavemax, image, fibers=fibers, width=7)
    
    # resampling on common finer wavelength grid
    flux, ivar, wave = resample_boxcar_frame(boxcar_flux, boxcar_ivar, boxcar_wave, oversampling=2)
    
    # median flux used as internal spectral reference
    mflux=np.median(flux,axis=0)

    # measure y shifts 
    return compute_dy_from_spectral_cross_correlations_of_frame(flux=flux, ivar=ivar, wave=wave, xcoef=xcoef, ycoef=ycoef, wavemin=wavemin, wavemax=wavemax, reference_flux = mflux , n_wavelength_bins = degyy+2)
    

def compute_dx_from_cross_dispersion_profiles(xcoef,ycoef,wavemin,wavemax, image, fibers=None, width=7,deg=2) :
    """
    Measure x offsets from a preprocessed image and a trace set 
    
    Args:
        xcoef : 2D np.array of shape (nfibers,ncoef) containing Legendre coefficents for each fiber to convert wavelenght to XCCD
        ycoef : 2D np.array of shape (nfibers,ncoef) containing Legendre coefficents for each fiber to convert wavelenght to YCCD
        wavemin : float
        wavemax : float. wavemin and wavemax are used to define a reduced variable legx(wave,wavemin,wavemax)=2*(wave-wavemin)/(wavemax-wavemin)-1
                  used to compute the traces, xccd=legval(legx(wave,wavemin,wavemax),xtrace[fiber]) 
        image : DESI preprocessed image object

    Optional:   
        fibers : 1D np.array of int (default is all fibers, the first fiber is always = 0)
        width  : extraction boxcar width, default is 5
        deg    : degree of polynomial fit as a function of y, only used to find and mask outliers
    
    Returns:
        x  : 1D array of x coordinates on CCD (axis=1 in numpy image array, AXIS=0 in FITS, cross-dispersion axis = fiber number direction) 
        y  : 1D array of y coordinates on CCD (axis=0 in numpy image array, AXIS=1 in FITS, wavelength dispersion axis)        
        dx : 1D array of shifts along x coordinates on CCD
        ex : 1D array of uncertainties on dx 
        fiber : 1D array of fiber ID (first fiber = 0) 
        wave  : 1D array of wavelength
    """
    log=get_logger()
    log.info("Starting compute_dx_from_cross_dispersion_profiles ...")
    
    if fibers is None :
        fibers = np.arange(psf.nspec)
    
    log.info("wavelength range : [%f,%f]"%(wavemin,wavemax))
    
    if image.mask is not None :
        image.ivar *= (image.mask==0)

        
    #   Variance based on inverse variance's size
    var    = np.zeros(image.ivar.shape)

    #   Applying a mask that keeps positive value to get the Variance by inversing the inverse variance.
    
    n0 = image.pix.shape[0]
    n1 = image.pix.shape[1]
    
    y  = np.arange(n0)
    xx = np.tile(np.arange(n1),(n0,1))
    hw = width//2
    
    ncoef=ycoef.shape[1]
    twave=np.linspace(wavemin, wavemax, ncoef+2)

    ox=np.array([])
    oy=np.array([])
    odx=np.array([])
    oex=np.array([])
    of=np.array([])
    ol=np.array([])
    
    for f,fiber in enumerate(fibers) :
        log.info("computing dx for fiber #%03d"%fiber)
        y_of_wave     = legval(legx(twave, wavemin, wavemax), ycoef[fiber])
        coef          = legfit(legx(y_of_wave, 0, n0), twave, deg=ncoef) # add one deg
        twave         = legval(legx(np.arange(n0).astype(float), 0, n0), coef)
        x_of_y        = legval(legx(twave, wavemin, wavemax), xcoef[fiber])
        x_of_y_int    = np.floor(x_of_y+0.5).astype(int)
        dx            = (xx.T-x_of_y).T
        mask=((xx.T>=x_of_y_int-hw)&(xx.T<=x_of_y_int+hw)).T
        swdx           = (dx[mask] * image.pix[mask] ).reshape((n0,width)).sum(-1)
        sw            = (image.pix[mask]).reshape((n0,width)).sum(-1)
        swy           = sw*y
        swx           = sw*x_of_y
        swl           = sw*twave

        # rebin
        rebin = 100
        sw  = sw[:(n0//rebin)*rebin].reshape(n0//rebin,rebin).sum(-1)
        swdx = swdx[:(n0//rebin)*rebin].reshape(n0//rebin,rebin).sum(-1)
        swx = swx[:(n0//rebin)*rebin].reshape(n0//rebin,rebin).sum(-1)
        swy = swy[:(n0//rebin)*rebin].reshape(n0//rebin,rebin).sum(-1)
        swl = swl[:(n0//rebin)*rebin].reshape(n0//rebin,rebin).sum(-1)


        sw[sw<0]       = 0        
        fdx            = swdx/(sw+(sw==0))
        fx             = swx/(sw+(sw==0))
        fy             = swy/(sw+(sw==0))
        fl             = swl/(sw+(sw==0))
        fex            = 1./np.sqrt(sw+(sw==0))
        
        good_fiber=True
        for loop in range(10) :

            if np.sum(sw>0) < deg+2 :
                good_fiber=False
                break

            try :
                c             = np.polyfit(fy,fdx,deg,w=sw)
                pol           = np.poly1d(c)
                chi2          = sw*(fdx-pol(fy))**2
                mchi2         = np.median(chi2[sw>0])
                sw /= mchi2
                bad           = chi2>25.*mchi2
                nbad          = np.sum(bad)
                sw[bad]       = 0.
            except LinAlgError :
                good_fiber=False
                break
            
            if nbad==0 :
                break
        
        
        # we return the original sample of offset values
        if good_fiber :
            ox  = np.append(ox,fx[sw>0])
            oy  = np.append(oy,fy[sw>0])
            odx = np.append(odx,fdx[sw>0])
            oex = np.append(oex,fex[sw>0])
            of = np.append(of,fiber*np.ones(fy[sw>0].size))
            ol = np.append(ol,fl[sw>0])
    
    return ox,oy,odx,oex,of,ol


def shift_ycoef_using_external_spectrum(psf,xcoef,ycoef,wavemin,wavemax,image,fibers,spectrum_filename,degyy=2,width=7) :
    """
    Measure y offsets (external wavelength calibration) from a preprocessed image , a PSF + trace set using a cross-correlation of boxcar extracted spectra
    and an external well-calibrated spectrum.
    The PSF shape is used to convolve the input spectrum. It could also be used to correct for the PSF asymetry (disabled for now).
    A relative flux calibration of the spectra is performed internally.
    
    Args:
        psf : specter PSF
        xcoef : 2D np.array of shape (nfibers,ncoef) containing Legendre coefficents for each fiber to convert wavelenght to XCCD
        ycoef : 2D np.array of shape (nfibers,ncoef) containing Legendre coefficents for each fiber to convert wavelenght to YCCD
        wavemin : float
        wavemax : float. wavemin and wavemax are used to define a reduced variable legx(wave,wavemin,wavemax)=2*(wave-wavemin)/(wavemax-wavemin)-1
                  used to compute the traces, xccd=legval(legx(wave,wavemin,wavemax),xtrace[fiber]) 
        image : DESI preprocessed image object
        fibers : 1D np.array of fiber indices
        spectrum_filename : path to input spectral file ( read with np.loadtxt , first column is wavelength (in vacuum and Angstrom) , second column in flux (arb. units)
    Optional:
        width  : int, extraction boxcar width, default is 7
        degyy  : int, degree of polynomial fit of shifts as a function of y, used to reject outliers.

    Returns:
        ycoef  : 2D np.array of same shape as input, with modified Legendre coefficents for each fiber to convert wavelenght to YCCD

    """
    log = get_logger()

    tmp=np.loadtxt(spectrum_filename).T
    ref_wave=tmp[0]
    ref_spectrum=tmp[1]
    log.info("read reference spectrum in %s with %d entries"%(spectrum_filename,ref_wave.size))

    log.info("rextract spectra with boxcar")   

    # boxcar extraction
    boxcar_flux, boxcar_ivar, boxcar_wave = boxcar_extraction(xcoef,ycoef,wavemin,wavemax, image, fibers=fibers, width=width)

    # resampling on common finer wavelength grid
    flux, ivar, wave = resample_boxcar_frame(boxcar_flux, boxcar_ivar, boxcar_wave, oversampling=2)
    
    # median flux used as internal spectral reference
    mflux=np.median(flux,axis=0)
    mivar=np.median(ivar,axis=0)*flux.shape[0]*(2./np.pi) # very appoximate !
    
    
    # trim ref_spectrum
    i=(ref_wave>=wave[0])&(ref_wave<=wave[-1])
    ref_wave=ref_wave[i]
    ref_spectrum=ref_spectrum[i]
    
    # check wave is linear or make it linear
    if np.abs((ref_wave[1]-ref_wave[0])-(ref_wave[-1]-ref_wave[-2]))>0.0001*(ref_wave[1]-ref_wave[0]) :
        log.info("reference spectrum wavelength is not on a linear grid, resample it")
        dwave = np.min(np.gradient(ref_wave))
        tmp_wave = np.linspace(ref_wave[0],ref_wave[-1],int((ref_wave[-1]-ref_wave[0])/dwave))
        ref_spectrum = resample_flux(tmp_wave, ref_wave , ref_spectrum)
        ref_wave = tmp_wave

    try :
        # compute psf at most significant line of ref_spectrum
        i=np.argmax(ref_spectrum)
        central_wave_for_psf_evaluation  = ref_wave[i]
        fiber_for_psf_evaluation = (boxcar_flux.shape[0]//2)
        dwave=ref_wave[i+1]-ref_wave[i]
        hw=int(3./dwave)+1 # 3A half width
        wave_range = ref_wave[i-hw:i+hw+1]
        x,y=psf.xy(fiber_for_psf_evaluation,wave_range)
        x=np.tile(x[hw]+np.arange(-hw,hw+1)*(y[-1]-y[0])/(2*hw+1),(y.size,1))
        y=np.tile(y,(2*hw+1,1)).T
        kernel2d=psf._value(x,y,fiber_for_psf_evaluation,central_wave_for_psf_evaluation)            
        kernel1d=np.sum(kernel2d,axis=1)
        log.info("convolve reference spectrum using PSF at fiber %d and wavelength %dA"%(fiber_for_psf_evaluation,central_wave_for_psf_evaluation))
        ref_spectrum=fftconvolve(ref_spectrum,kernel1d, mode='same')
    except :
        log.warning("couldn't convolve reference spectrum: %s %s"%(sys.exc_info()[0],sys.exc_info()[1]))
    
    
    
    # resample input spectrum
    log.info("resample convolved reference spectrum")
    ref_spectrum = resample_flux(wave, ref_wave , ref_spectrum)

    log.info("absorb difference of calibration")
    x=(wave-wave[wave.size//2])/50.
    kernel=np.exp(-x**2/2)
    f1=fftconvolve(mflux,kernel,mode='same')
    f2=fftconvolve(ref_spectrum,kernel,mode='same')
    scale=f1/f2
    ref_spectrum *= scale
    
    log.info("fit shifts on wavelength bins")
    # define bins
    n_wavelength_bins = degyy+4
    y_for_dy=np.array([])
    dy=np.array([])
    ey=np.array([])
    wave_for_dy=np.array([])
    for b in range(n_wavelength_bins) :
        wmin=wave[0]+((wave[-1]-wave[0])/n_wavelength_bins)*b
        if b<n_wavelength_bins-1 :
            wmax=wave[0]+((wave[-1]-wave[0])/n_wavelength_bins)*(b+1)
        else :
            wmax=wave[-1]
        ok=(wave>=wmin)&(wave<=wmax)
        sw= np.sum(mflux[ok]*(mflux[ok]>0))
        if sw==0 :
            continue
        dwave,err = compute_dy_from_spectral_cross_correlation(mflux[ok],wave[ok],ref_spectrum[ok],ivar=mivar[ok],hw=3.)
        bin_wave  = np.sum(mflux[ok]*(mflux[ok]>0)*wave[ok])/sw
        x,y=psf.xy(fiber_for_psf_evaluation,bin_wave)
        eps=0.1
        x,yp=psf.xy(fiber_for_psf_evaluation,bin_wave+eps)
        dydw=(yp-y)/eps
        
        if err*dydw<1 :
            dy=np.append(dy,-dwave*dydw)
            ey=np.append(ey,err*dydw)
            wave_for_dy=np.append(wave_for_dy,bin_wave)
            y_for_dy=np.append(y_for_dy,y)
            log.info("wave = %fA , y=%d, measured dwave = %f +- %f A"%(bin_wave,y,dwave,err))
    
    if False : # we don't need this for now
        try :
            log.info("correcting bias due to asymmetry of PSF")

            hw=5
            oversampling=4
            xx=np.tile(np.arange(2*hw*oversampling+1)-hw*oversampling,(2*hw*oversampling+1,1))/float(oversampling)
            yy=xx.T
            x,y=psf.xy(fiber_for_psf_evaluation,central_wave_for_psf_evaluation)
            prof=psf._value(xx+x,yy+y,fiber_for_psf_evaluation,central_wave_for_psf_evaluation)
            dy_asym_central = np.sum(yy*prof)/np.sum(prof)
            for i in range(dy.size) :
                x,y=psf.xy(fiber_for_psf_evaluation,wave_for_dy[i])
                prof=psf._value(xx+x,yy+y,fiber_for_psf_evaluation,wave_for_dy[i])
                dy_asym = np.sum(yy*prof)/np.sum(prof)
                log.info("y=%f, measured dy=%f , bias due to PSF asymetry = %f"%(y,dy[i],dy_asym-dy_asym_central))
                dy[i] -= (dy_asym-dy_asym_central)
        except :
            log.warning("couldn't correct for asymmetry of PSF: %s %s"%(sys.exc_info()[0],sys.exc_info()[1]))

    log.info("polynomial fit of shifts and modification of PSF ycoef")
    # pol fit
    coef = np.polyfit(wave_for_dy,dy,degyy,w=1./ey**2)
    pol  = np.poly1d(coef)

    for i in range(dy.size) :
        log.info("wave=%fA y=%f, measured dy=%f+-%f , pol(wave) = %f"%(wave_for_dy[i],y_for_dy[i],dy[i],ey[i],pol(wave_for_dy[i])))

    log.info("apply this to the PSF ycoef")
    wave = np.linspace(wavemin,wavemax,100)
    dy   = pol(wave)
    dycoef = legfit(legx(wave,wavemin,wavemax),dy,deg=ycoef.shape[1]-1)
    for fiber in range(ycoef.shape[0]) :
        ycoef[fiber] += dycoef

    return ycoef


# end of routines for cross-correlation method for trace shifts  

# beginning of routines for forward model method for trace shifts  

def compute_fiber_bundle_trace_shifts_using_psf(fibers,line,psf,image,maxshift=2.) :
    """
    Computes trace shifts along x and y from a preprocessed image, a PSF (with trace coords), and a given emission line,
    by doing a forward model of the image.
        
    Args:
        fibers : 1D array with list of fibers
        line : float, wavelength of an emission line (in Angstrom)
        psf  : specter psf object
        image : DESI preprocessed image object
    Optional:
        maxshift : float maximum shift in pixels for 2D chi2 scan
    
    Returns:
        x  : 1D array of x coordinates on CCD (axis=1 in numpy image array, AXIS=0 in FITS, cross-dispersion axis = fiber number direction) 
        y  : 1D array of y coordinates on CCD (axis=0 in numpy image array, AXIS=1 in FITS, wavelength dispersion axis)        
        dx : 1D array of shifts along x coordinates on CCD
        dy : 1D array of shifts along y coordinates on CCD
        sx : 1D array of uncertainties on dx 
        sy : 1D array of uncertainties on dy      
    """
    log=get_logger()
    #log.info("compute_fiber_bundle_offsets fibers={} line={}".format(fibers,line))

    # get central coordinates of bundle for interpolation of offsets on CCD
    x,y = psf.xy([int(np.median(fibers)),],line)
    

    try : 
        nfibers=len(fibers)
        
        # compute stamp coordinates
        xstart=None
        xstop=None
        ystart=None
        ystop=None
        xs=[]
        ys=[]
        pix=[]
        xx=[]
        yy=[]
        
        for fiber in fibers :
            txs,tys,tpix = psf.xypix(fiber,line)
            xs.append(txs)
            ys.append(tys)
            pix.append(tpix)
            if xstart is None :
                xstart =txs.start
                xstop  =txs.stop
                ystart =tys.start
                ystop  =tys.stop
            else :
                xstart =min(xstart,txs.start)
                xstop  =max(xstop,txs.stop)
                ystart =min(ystart,tys.start)
                ystop  =max(ystop,tys.stop)

        # load stamp data, with margins to avoid problems with shifted psf
        margin=int(maxshift)+1
        stamp=np.zeros((ystop-ystart+2*margin,xstop-xstart+2*margin))
        stampivar=np.zeros(stamp.shape)
        stamp[margin:-margin,margin:-margin]=image.pix[ystart:ystop,xstart:xstop]
        stampivar[margin:-margin,margin:-margin]=image.ivar[ystart:ystop,xstart:xstop]


        # will use a fixed footprint despite changes of psf stamps
        # so that chi2 always based on same data set
        footprint=np.zeros(stamp.shape)   
        for i in range(nfibers) :
            footprint[margin-ystart+ys[i].start:margin-ystart+ys[i].stop,margin-xstart+xs[i].start:margin-xstart+xs[i].stop]=1

        #plt.imshow(footprint) ; plt.show() ; sys.exit(12)

        # define grid of shifts to test
        res=0.5
        nshift=int(maxshift/res)
        dx=res*np.tile(np.arange(2*nshift+1)-nshift,(2*nshift+1,1))
        dy=dx.T
        original_shape=dx.shape
        dx=dx.ravel()
        dy=dy.ravel()
        chi2=np.zeros(dx.shape)

        A=np.zeros((nfibers,nfibers))
        B=np.zeros((nfibers))
        mods=np.zeros(np.zeros(nfibers).shape+stamp.shape)
        
        debugging=False

        if debugging : # FOR DEBUGGING KEEP MODELS            
            models=[]


        # loop on possible shifts
        # refit fluxes and compute chi2
        for d in range(len(dx)) :
            # print(d,dx[d],dy[d])
            A *= 0
            B *= 0
            mods *= 0

            for i,fiber in enumerate(fibers) :

                # apply the PSF shift
                psf._cache={} # reset cache !!
                psf.coeff['X']._coeff[fiber][0] += dx[d]
                psf.coeff['Y']._coeff[fiber][0] += dy[d]

                # compute pix and paste on stamp frame
                xx, yy, pix = psf.xypix(fiber,line)
                mods[i][margin-ystart+yy.start:margin-ystart+yy.stop,margin-xstart+xx.start:margin-xstart+xx.stop]=pix

                # undo the PSF shift
                psf.coeff['X']._coeff[fiber][0] -= dx[d]
                psf.coeff['Y']._coeff[fiber][0] -= dy[d]

                B[i] = np.sum(stampivar*stamp*mods[i])
                for j in range(i+1) :
                    A[i,j] = np.sum(stampivar*mods[i]*mods[j]) 
                    if j!=i :
                        A[j,i] = A[i,j]
            Ai=np.linalg.inv(A)
            flux=Ai.dot(B)
            model=np.zeros(stamp.shape)
            for i in range(nfibers) :
                model += flux[i]*mods[i]
            chi2[d]=np.sum(stampivar*(stamp-model)**2)
            if debugging :
                models.append(model)
            
        if debugging :
            schi2=chi2.reshape(original_shape).copy() # FOR DEBUGGING
            sdx=dx.copy()
            sdy=dy.copy()
        
        # find minimum chi2 grid point
        k   = chi2.argmin()
        j,i = np.unravel_index(k, ((2*nshift+1),(2*nshift+1)))
        #print("node dx,dy=",dx.reshape(original_shape)[j,i],dy.reshape(original_shape)[j,i])
        
        # cut a region around minimum
        delta=1
        istart=max(0,i-delta)
        istop=min(2*nshift+1,i+delta+1)
        jstart=max(0,j-delta)
        jstop=min(2*nshift+1,j+delta+1)
        chi2=chi2.reshape(original_shape)[jstart:jstop,istart:istop].ravel()
        dx=dx.reshape(original_shape)[jstart:jstop,istart:istop].ravel()
        dy=dy.reshape(original_shape)[jstart:jstop,istart:istop].ravel()    
        # fit 2D polynomial of deg2
        m = np.array([dx*0+1, dx, dy, dx**2, dy**2, dx*dy ]).T
        c, r, rank, s = np.linalg.lstsq(m, chi2)
        if c[3]>0 and c[4]>0 :
            # get minimum
            # dchi2/dx=0 : c[1]+2*c[3]*dx+c[5]*dy = 0
            # dchi2/dy=0 : c[2]+2*c[4]*dy+c[5]*dx = 0
            a=np.array([[2*c[3],c[5]],[c[5],2*c[4]]])
            b=np.array([c[1],c[2]])
            t=-np.linalg.inv(a).dot(b)
            dx=t[0]
            dy=t[1]
            sx=1./np.sqrt(c[3])
            sy=1./np.sqrt(c[4])
            #print("interp dx,dy=",dx,dy)
            
            if debugging : # FOR DEBUGGING
                import matplotlib.pyplot as plt
                plt.figure()
                plt.subplot(2,2,1,title="chi2")
                plt.imshow(schi2,extent=(-nshift*res,nshift*res,-nshift*res,nshift*res),origin=0,interpolation="nearest")            
                plt.plot(dx,dy,"+",color="white",ms=20)
                plt.xlabel("x")
                plt.ylabel("y")
                plt.subplot(2,2,2,title="data")
                plt.imshow(stamp*footprint,origin=0,interpolation="nearest")
                plt.grid()
                k0=np.argmin(sdx**2+sdy**2)
                plt.subplot(2,2,3,title="original psf")
                plt.imshow(models[k0],origin=0,interpolation="nearest")            
                plt.grid()
                plt.subplot(2,2,4,title="shifted psf")
                plt.imshow(models[k],origin=0,interpolation="nearest")
                plt.grid()
                plt.show()
                
        else :
            log.warning("fit failed (bad chi2 surf.) for fibers [%d:%d] line=%dA"%(fibers[0],fibers[-1]+1,int(line)))
            dx=0.
            dy=0.
            sx=10.
            sy=10.
    except LinAlgError :
        log.warning("fit failed (masked or missing data) for fibers [%d:%d] line=%dA"%(fibers[0],fibers[-1]+1,int(line)))
        dx=0.
        dy=0.
        sx=10.
        sy=10.
    
    return x,y,dx,dy,sx,sy



    
def compute_dx_dy_using_psf(psf,image,fibers,lines) :
    """
    Computes trace shifts along x and y from a preprocessed image, a PSF (with trace coords), and a set of emission lines,
    by doing a forward model of the image.
    Calls compute_fiber_bundle_trace_shifts_using_psf.
        
    Args:
        psf  : specter psf object
        image : DESI preprocessed image object
        fibers : 1D array with list of fibers
        lines : 1D array of wavelength of emission lines (in Angstrom)
    
    Returns:
        x  : 1D array of x coordinates on CCD (axis=1 in numpy image array, AXIS=0 in FITS, cross-dispersion axis = fiber number direction) 
        y  : 1D array of y coordinates on CCD (axis=0 in numpy image array, AXIS=1 in FITS, wavelength dispersion axis)        
        dx : 1D array of shifts along x coordinates on CCD
        dy : 1D array of shifts along y coordinates on CCD
        sx : 1D array of uncertainties on dx 
        sy : 1D array of uncertainties on dy
        fiber : 1D array of fiber ID
        wave  : 1D array of wavelength

    """
    log = get_logger()
    
    nlines=len(lines)
    nfibers=len(fibers)
    
    log.info("computing spots coordinates and define bundles")
    x=np.zeros((nfibers,nlines))
    y=np.zeros((nfibers,nlines))
    
    # load expected spots coordinates 
    for fiber in range(nfibers) :
        for l,line in enumerate(lines) :
            x[fiber,l],y[fiber,l] = psf.xy(fiber,line)

    bundle_fibers=[]
    bundle_xmin=[]
    bundle_xmax=[]
    xwidth=9.
    bundle_xmin.append(x[0,nlines//2]-xwidth/2)
    bundle_xmax.append(x[0,nlines//2]+xwidth/2)
    bundle_fibers.append([0,])
    
    
    for fiber in range(1,nfibers) :
        tx=x[fiber,nlines//2]
        found=False
        for b in range(len(bundle_fibers)) :
            if tx+xwidth/2 >= bundle_xmin[b] and tx-xwidth/2 <= bundle_xmax[b] :
                found=True
                bundle_fibers[b].append(fiber)
                bundle_xmin[b]=min(bundle_xmin[b],tx-xwidth/2)
                bundle_xmax[b]=max(bundle_xmax[b],tx+xwidth/2)
                break
        if not found :
            bundle_fibers.append([fiber,])
            bundle_xmin.append(tx-xwidth/2)
            bundle_xmax.append(tx+xwidth/2)
    
    log.info("measure offsets dx dy per bundle ({}) and spectral line ({})".format(len(bundle_fibers),len(lines)))

    wave_xy=np.array([])  # line
    fiber_xy=np.array([])  # central fiber in bundle
    x=np.array([])  # central x in bundle at line wavelength
    y=np.array([])  # central x in bundle at line wavelength
    dx=np.array([]) # measured offset along x
    dy=np.array([]) # measured offset along y
    ex=np.array([]) # measured offset uncertainty along x
    ey=np.array([]) # measured offset uncertainty along y
    
    for b in range(len(bundle_fibers)) :
        for l,line in enumerate(lines) :
            tx,ty,tdx,tdy,tex,tey = compute_fiber_bundle_trace_shifts_using_psf(fibers=bundle_fibers[b],psf=psf,image=image,line=line)
            log.info("fibers [%d:%d] %dA dx=%4.3f+-%4.3f dy=%4.3f+-%4.3f"%(bundle_fibers[b][0],bundle_fibers[b][-1]+1,int(line),tdx,tex,tdy,tey))
            if tex<1. and tey<1. :                
                wave_xy=np.append(wave_xy,line)
                fiber_xy=np.append(fiber_xy,int(np.median(bundle_fibers[b])))
                x=np.append(x,tx)
                y=np.append(y,ty)
                dx=np.append(dx,tdx)
                dy=np.append(dy,tdy)
                ex=np.append(ex,tex)
                ey=np.append(ey,tey)
    return x,y,dx,ex,dy,ey,fiber_xy,wave_xy

# end of routines for forward model method    

def monomials(x,y,degx,degy) :
    """
    Computes monomials as a function of x and y of a 2D polynomial of degrees degx and degy
    
    Args:
        x : ND array
        y : ND array of same shape as x
        degx : int (>=0), polynomial degree along x
        degy : int (>=0), polynomial degree along y
    
    Returns :
       monomials : ND array of shape ( (degx+1)*(degy+1) , x shape )
        
    """
    M=[]
    for i in range(degx+1) :
        for j in range(degy+1) :
            M.append(x**i*y**j)
    return np.array(M)
    
def polynomial_fit(z,ez,xx,yy,degx,degy) :
    """
    Computes and 2D polynomial fit of z as a function of (x,y) of degrees degx and degy
    
    Args:
        z : ND array 
        ez : ND array of same shape as z, uncertainties on z
        x : ND array of same shape as z
        y : ND array of same shape as z
        degx : int (>=0), polynomial degree along x
        degy : int (>=0), polynomial degree along y
    
    Returns:
        coeff : 1D array of size (degx+1)*(degy+1) with polynomial coefficients (as defined by routine monomials)
        covariance : 2D array of covariance of coeff 
        error_floor : float , extra uncertainty needed to get chi2/ndf=1 
        polval : ND array of same shape as z with values of pol(x,y) 
        mask : ND array of same shape as z indicating the masked data points in the fit
        
    """
    M=monomials(x=xx,y=yy,degx=degx,degy=degy)
    
    a_large_error = 1.e4
    ez[ez>1]= a_large_error # totally deweight unmeasured data
    
    error_floor=0.002 # pix
    
    npar=M.shape[0]
    A=np.zeros((npar,npar))
    B=np.zeros((npar))
    
    mask=(ez<a_large_error)
    for loop in range(100) : # loop to increase errors
        
        w=1./(ez**2+error_floor**2)
        w[mask==0]=0.
        
        A *= 0.
        B *= 0.
        for k in range(npar) :
            B[k]=np.sum(w*z*M[k])
            for l in range(k+1) :
                A[k,l]=np.sum(w*M[k]*M[l])
                if l!=k : A[l,k]=A[k,l]
        coeff=cholesky_solve(A,B)
        polval = M.T.dot(coeff)
        
        # compute rchi2 with median
        ndata=np.sum(w>0)
        rchi2=1.4826*np.median(np.sqrt(w)*np.abs(z-polval))*ndata/float(ndata-npar)
        # std chi2 
        rchi2_std = np.sum(w*(z-polval)**2)/(ndata-npar)
        #print("#%d rchi2=%f rchi2_std=%f ngood=%d nbad=%d error floor=%f"%(loop,rchi2,rchi2_std,ndata,np.sum(w==0),error_floor))
        
        # reject huge outliers
        nbad=0
        rvar=w*(z-polval)**2
        worst=np.argmax(rvar)
        if rvar[worst] > 25*max(rchi2,1.2) : # cap rchi2 if starting point is very bad
            #print("remove one bad measurement at %2.1f sigmas"%np.sqrt(rvar[worst]))
            mask[worst]=0
            nbad=1
        
        if rchi2>1 :
            if nbad==0 or loop>5 :
                error_floor+=0.002
        
        if rchi2<=1. and nbad==0 :
            break
    
    # rerun chol. solve to get covariance
    coeff,covariance=cholesky_solve_and_invert(A,B)
        
        
    return coeff,covariance,error_floor,polval,mask
    
def recompute_legendre_coefficients(xcoef,ycoef,wavemin,wavemax,degxx,degxy,degyx,degyy,dx_coeff,dy_coeff) :
    """
    Modifies legendre coefficients of an input trace set using polynomial coefficents (as defined by the routine monomials)
    
    Args:
        xcoef : 2D np.array of shape (nfibers,ncoef) containing Legendre coefficents for each fiber to convert wavelenght to XCCD
        ycoef : 2D np.array of shape (nfibers,ncoef) containing Legendre coefficents for each fiber to convert wavelenght to YCCD
        wavemin : float
        wavemax : float. wavemin and wavemax are used to define a reduced variable legx(wave,wavemin,wavemax)=2*(wave-wavemin)/(wavemax-wavemin)-1
                  used to compute the traces, xccd=legval(legx(wave,wavemin,wavemax),xtrace[fiber]) 
        degxx : int, degree of polynomial for x shifts as a function of x (x is axis=1 in numpy image array, AXIS=0 in FITS, cross-dispersion axis = fiber number direction)
        degxy : int, degree of polynomial for x shifts as a function of y (y is axis=0 in numpy image array, AXIS=1 in FITS, wavelength dispersion axis)
        degyx : int, degree of polynomial for y shifts as a function of x
        degyy : int, degree of polynomial for y shifts as a function of y
        dx_coeff : 1D np.array of polynomial coefficients of size (degxx*degxy) as defined by the routine monomials.
        dy_coeff : 1D np.array of polynomial coefficients of size (degyx*degyy) as defined by the routine monomials.
        
    Returns:
        xcoef : 2D np.array of shape (nfibers,ncoef) with modified Legendre coefficents
        ycoef : 2D np.array of shape (nfibers,ncoef) with modified Legendre coefficents
    """
    wave=np.linspace(wavemin,wavemax,100)
    nfibers=xcoef.shape[0]
    rw=legx(wave,wavemin,wavemax)
    for fiber in range(nfibers) :
        x = legval(rw,xcoef[fiber])
        y = legval(rw,ycoef[fiber])
                
        m=monomials(x,y,degxx,degxy)
        dx=m.T.dot(dx_coeff)
        xcoef[fiber]=legfit(rw,x+dx,deg=xcoef.shape[1]-1)
        
        m=monomials(x,y,degyx,degyy)
        dy=m.T.dot(dy_coeff)
        ycoef[fiber]=legfit(rw,y+dy,deg=ycoef.shape[1]-1)
    return xcoef,ycoef
