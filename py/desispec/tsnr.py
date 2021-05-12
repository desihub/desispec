import os
import numpy as np
import astropy.io.fits as fits
import copy
import glob
import json
import numpy as np
import yaml

from pkg_resources import resource_filename
from desispec.io.spectra import Spectra
from astropy.convolution import convolve, Box1DKernel
from scipy.interpolate import RectBivariateSpline
from specter.psf.gausshermite  import  GaussHermitePSF
from desispec.calibfinder import findcalibfile
from desiutil.log import get_logger
from scipy.optimize import minimize
from desiutil.dust import ext_odonnell
from desispec.fiberfluxcorr import psf_to_fiber_flux_correction
from desimodel.fastfiberacceptance import FastFiberAcceptance
from desimodel.io import load_platescale
from desispec.io.meta import findfile
from astropy import units
from astropy import constants as const
from desimodel.io import load_desiparams

def dust_transmission(wave,ebv):
    Rv = 3.1
    extinction = ext_odonnell(wave,Rv=Rv)
    return 10**(-Rv*ebv[:,None]*extinction[None,:]/2.5)

def get_ensemble(dirpath, bands, smooth=125):
    '''
    Function that takes a frame object and a bitmask and
    returns ivar (and optionally mask) array(s) that have fibers with
    offending bits in fibermap['FIBERSTATUS'] set to
    0 in ivar and optionally flips a bit in mask.

    Args:
        dirpath: path to the dir. with ensemble dflux files.
        bands:  bands to expect, typically [BRZ] - case ignored.

    Options:
        smooth:  Further convolve the residual ensemble flux.

    Returns:
        Dictionary with keys labelling each tracer (bgs, lrg, etc.) for which each value
        is a Spectra class instance with wave, flux for BRZ arms.  Note flux is the high
        frequency residual for the ensemble.  See doc. 4723.
    '''
    paths = glob.glob(dirpath + '/tsnr-ensemble-*.fits')

    wave = {}
    flux = {}
    ivar = {}
    mask = {}
    res  = {}

    ensembles  = {}

    for path in paths:
        tracer = path.split('/')[-1].split('-')[2].replace('.fits','')
        dat    = fits.open(path)

        for band in bands:
            wave[band] = dat['WAVE_{}'.format(band.upper())].data
            flux[band] = dat['DFLUX_{}'.format(band.upper())].data
            ivar[band] = 1.e99 * np.ones_like(flux[band])

            # 125: 100. A in 0.8 pixel.
            if smooth > 0:
                flux[band] = convolve(flux[band][0,:], Box1DKernel(smooth), boundary='extend')
                flux[band] = flux[band].reshape(1, len(flux[band]))

        ensembles[tracer] = Spectra(bands, wave, flux, ivar)

    return  ensembles

def read_nea(path):
    '''
    Read a master noise equivalent area [sq. pixel] file.

    input:
        path: path to a master nea file for a given camera, e.g. b0.

    returns:
        nea: 2D split object to be evaluated at (fiber, wavelength)
        angperpix:  2D split object to be evaluated at (fiber, wavelength),
                    yielding angstrom per pixel.
    '''

    with fits.open(path, memmap=False) as fx:
        wave=fx['WAVELENGTH'].data
        angperpix=fx['ANGPERPIX'].data
        nea=fx['NEA'].data

    fiber = np.arange(len(nea))

    nea = RectBivariateSpline(fiber, wave, nea)
    angperpix = RectBivariateSpline(fiber, wave, angperpix)

    return  nea, angperpix

def fb_rdnoise(fibers, frame, psf):
    '''
    Approximate the readnoise for a given fiber (on a given camera) for the
    wavelengths present in frame. wave.

    input:
        fibers: e.g. np.arange(500) to index fiber.
        frame:  frame instance for the given camera.
        psf:  corresponding psf instance.

    returns:
        rdnoise: (nfiber x nwave) array with the estimated readnosie.  Same
                 units as OBSRDNA, e.g. ang per pix.
    '''

    ccdsizes = np.array(frame.meta['CCDSIZE'].split(',')).astype(np.float)

    xtrans = ccdsizes[0] / 2.
    ytrans = ccdsizes[1] / 2.

    rdnoise = np.zeros_like(frame.flux)

    for ifiber in fibers:
        wave_lim = psf.wavelength(ispec=ifiber, y=ytrans)
        x = psf.x(ifiber, wave_lim)

        # A | C.
        if x < xtrans:
            rdnoise[ifiber, frame.wave <  wave_lim] = frame.meta['OBSRDNA']
            rdnoise[ifiber, frame.wave >= wave_lim] = frame.meta['OBSRDNC']

        # B | D
        else:
            rdnoise[ifiber, frame.wave <  wave_lim] = frame.meta['OBSRDNB']
            rdnoise[ifiber, frame.wave >= wave_lim] = frame.meta['OBSRDND']

    return rdnoise

def var_model(rdnoise_sigma, npix_1d, angperpix, angperspecbin, fiberflat, skymodel, alpha=1.0, components=False):
    '''
    Evaluate a model for the 1D spectral flux variance, e.g. quadrature sum of readnoise and sky components.

    input:
        rdnoise_sigma:
        npix_1d:  equivalent to (1D) nea.
        angperpix:  Angstroms per pixel.
        angperspecbin: Angstroms per bin.
        fiberflat: fiberflat instance
        skymodel: Sky instance.
        alpha: empirical weighting of the rdnoise term to e.g. better fit sky fibers per exp. cam.
        components:  if True, return tuple of individual contributions to the variance.  Else return variance.

    returns:
        nfiber x nwave array of the expected variance.
    '''

    # the extraction is performed with a wavelength bin of width = angperspecbin
    # so the effective number of CCD pixels corresponding to a spectral bin width is
    npix_2d = npix_1d * (angperspecbin / angperpix)

    # then, the extracted flux per specbin is converted to an extracted flux per A, so
    # the variance has to be divided by the square of the conversion factor = angperspecbin**2

    rdnoise_variance = rdnoise_sigma**2 * npix_2d / angperspecbin**2

    # It was verified that this variance has to be increased by about 10% to match the
    # inverse variance reported in the frame files of a zero exposure (exptime=0).
    # However the correction factor (alpha) can be larger when fitted on sky fibers
    # because the precomputed effective noise equivalent number of pixels (npix_1d)
    # is valid only when the Poisson noise is negligible. It increases with the spectral flux.

    if components:
        return (alpha * rdnoise_variance, fiberflat.fiberflat * np.abs(skymodel.flux))

    else:
        return alpha * rdnoise_variance + fiberflat.fiberflat * np.abs(skymodel.flux)

def var_tracer(tracer, frame, angperspecbin, fiberflat, fluxcalib, fiber_diameter_arcsec=1.52, exposure_seeing_fwhm=1.1):
    '''
    Source Poisson term to the model ivar, following conventions defined at:
        https://desi.lbl.gov/trac/wiki/SurveyOps/SurveySpeed.  

    See also:
        https://github.com/desihub/desispec/blob/master/py/desispec/efftime.py

    '''
    log = get_logger()

    fiber_area_arcsec2 = np.pi*(fiber_diameter_arcsec/2.)**2
    
    if tracer == 'bgs':
        # Nominal fiberloss dependence on seeing.  Assumes zero offset. 
        fiberfrac = np.exp(0.0341 * np.log(exposure_seeing_fwhm)**3 -0.3611 * np.log(exposure_seeing_fwhm)**2 -0.7175 * np.log(exposure_seeing_fwhm) -1.5643)

        # Note: neglects transparency & EBV corrections. 
        nominal = 15.8                    # r=19.5 [nanomaggie].    

    elif tracer == 'backup':
        fiberfrac = np.exp(0.0989 * np.log(exposure_seeing_fwhm)**3 -0.5588 * np.log(exposure_seeing_fwhm)**2 -0.9708 * np.log(exposure_seeing_fwhm) -0.4473)
        nominal = 27.5                    # r=18.9 [nanomaggie].
        
    else:
        # No source poisson term otherwise. 
        nominal = 0.0                     # [nanomaggie]. 
        fiberfrac = 1.0

    nominal *= fiberfrac
    nominal /= fiber_area_arcsec2         # [nMg / ''].

    log.info('Including {} poisson source var of {:.6f} [nMg / sq. arcsec.]'.format(tracer, nominal))
    
    nominal *= 1.e-9                      # [ Mg / ''].                                                                                                                                                                                     
    nominal /= (1.e23 / const.c.value / 1.e10 / 3631.)
    nominal /= (frame.wave)**2.           # [ergs/s/cm2/A].                                                                                                                                                                            

    nominal *= 1.e17                      # [1.e-17 ergs/s/cm2/A].  
    
    nominal  = fluxcalib.calib * nominal  # [e/A]
    
    nominal *= angperspecbin  	          # [e/specbin].                                                                                                                                                                                    
    nominal *= fiberflat.fiberflat

    log.info('Including {} poisson source var of {:.6e} [e/specbin]'.format(tracer, np.median(nominal)))
    
    return  nominal

def gen_mask(frame, skymodel, hw=5.):
    """
    Generate a mask for the alpha computation, masking out bright sky lines.
    Args:
        frame : uncalibrated Frame object for one camera
        skymodel : SkyModel object
        hw : (optional) float, half width of mask around sky lines in A
    Returns an array of same shape as frame, here mask=1 is good, 0 is bad
    """
    log = get_logger()

    maskfactor = np.ones_like(frame.mask, dtype=np.float)
    maskfactor[frame.mask > 0] = 0.0

    # https://github.com/desihub/desispec/blob/294cfb66428aa8be3797fd046adbd0a2267c4409/py/desispec/sky.py#L1267
    skyline=np.array([5199.4,5578.4,5656.4,5891.4,5897.4,6302.4,6308.4,6365.4,6500.4,6546.4,\
                      6555.4,6618.4,6663.4,6679.4,6690.4,6765.4,6831.4,6836.4,6865.4,6925.4,\
                      6951.4,6980.4,7242.4,7247.4,7278.4,7286.4,7305.4,7318.4,7331.4,7343.4,\
                      7360.4,7371.4,7394.4,7404.4,7440.4,7526.4,7714.4,7719.4,7752.4,7762.4,\
                      7782.4,7796.4,7810.4,7823.4,7843.4,7855.4,7862.4,7873.4,7881.4,7892.4,\
                      7915.4,7923.4,7933.4,7951.4,7966.4,7982.4,7995.4,8016.4,8028.4,8064.4,\
                      8280.4,8284.4,8290.4,8298.4,8301.4,8313.4,8346.4,8355.4,8367.4,8384.4,\
                      8401.4,8417.4,8432.4,8454.4,8467.4,8495.4,8507.4,8627.4,8630.4,8634.4,\
                      8638.4,8652.4,8657.4,8662.4,8667.4,8672.4,8677.4,8683.4,8763.4,8770.4,\
                      8780.4,8793.4,8829.4,8835.4,8838.4,8852.4,8870.4,8888.4,8905.4,8922.4,\
                      8945.4,8960.4,8990.4,9003.4,9040.4,9052.4,9105.4,9227.4,9309.4,9315.4,\
                      9320.4,9326.4,9340.4,9378.4,9389.4,9404.4,9422.4,9442.4,9461.4,9479.4,\
                      9505.4,9521.4,9555.4,9570.4,9610.4,9623.4,9671.4,9684.4,9693.4,9702.4,\
                      9714.4,9722.4,9740.4,9748.4,9793.4,9802.4,9814.4,9820.4])

    maskfactor *= (skymodel.ivar > 0.0)
    maskfactor *= (frame.ivar > 0.0)

    if hw > 0.0:
        log.info('TSNR Masking bright lines in alpha calc. (half width: {:.3f})'.format(hw))

        for line in skyline :
            if line<=frame.wave[0] or line>=frame.wave[-1]:
                continue

            ii=np.where((frame.wave>=line-hw)&(frame.wave<=line+hw))[0]

            maskfactor[:,ii]=0.0

    # Mask collimator, [4300-4500A]
    ii=np.where((frame.wave>=4300.)&(frame.wave<=4500.))[0]
    maskfactor[:,ii]=0.0

    return maskfactor

def calc_alpha(frame, fibermap, rdnoise_sigma, npix_1d, angperpix, angperspecbin, fiberflat, skymodel):
    '''
    Model Var = alpha * rdnoise component + sky.

    Calculate the best-fit alpha using the sky fibers
    available to the frame.

    input:
        frame: desispec frame instance (should be uncalibrated, i.e. e/A).
        fibermap: desispec fibermap instance.
        rdnoise_sigma:  e.g. RDNOISE value per Quadrant (float).
        npix_1d:  equivalent to 1D nea [pixels], calculated using read_nea().
        angperpix:  angstroms per pixel (float),
        fiberflat: desispec fiberflat instance.
        skymodel: desispec Sky instance.
        alpha:  nuisanve parameter to reweight rdnoise vs sky contribution to variance (float).
        components:  if True, return individual contributions to variance, else return total variance.

    returns:
       alpha:  nuisance parameter to reweight rdnoise vs sky contribution to variance (float), obtained
               as the best fit to the uncalibrated sky fibers VAR.
    '''
    log = get_logger()
    sky_indx = np.where(fibermap['OBJTYPE'] == 'SKY')[0]
    rd_var, sky_var = var_model(rdnoise_sigma, npix_1d, angperpix, angperspecbin, fiberflat, skymodel, alpha=1.0, components=True)

    maskfactor = gen_mask(frame, skymodel)
    maskfactor = maskfactor[sky_indx,:]

    def calc_alphavar(alpha):
        return alpha * rd_var[sky_indx,:] + sky_var[sky_indx,:]

    def alpha_X2(alpha):
        _var = calc_alphavar(alpha)
        _ivar =  1. / _var
        X2 = np.abs(frame.ivar[sky_indx,:] - _ivar)

        return np.sum(X2 * maskfactor)

    res = minimize(alpha_X2, x0=[1.])
    alpha = res.x[0]

    #- From JG PR #1164:
    # Noisy values of alpha can occur for observations dominated by sky noise
    # where it is not possible to calibrated the read noise. For those
    # exposures, the precise value of alpha does not impact the SNR estimation.
    if alpha < 0.8 :
        log.warning(f'tSNR forcing best fit alpha = {alpha:.4f} to 0.8')
        alpha = 0.8

    return alpha

def calc_tsnr_fiberfracs(fibermap, etc_fiberfracs, no_offsets=False):
    '''
    Nominal fiberfracs for effective depths. See:
    https://desi.lbl.gov/trac/wiki/SurveyOps/SurveySpeed

    Args:
       fibermap:
       etc_fiberfracs:

    Returns:
       dict of the nominal fiberloss of given type and seeing. 
    '''
    
    log = get_logger()

    for k in ["FIBER_X","FIBER_Y"] :
        if k not in fibermap.dtype.names :
            log.warning("no column '{}' in fibermap, cannot do the tsnr fiberfrac correction, returning 1".format(k))
            return np.ones(len(fibermap))

    fa = FastFiberAcceptance()
        
    # Compute the effective seeing.
    exposure_seeing_fwhm = fa.psf_seeing_fwhm(etc_fiberfracs['psf']) # [microns]

    fiber_params = load_desiparams()['fibers']
    fiber_dia = fiber_params['diameter_um'] # 107 um.
    fiber_dia_asec = fiber_params['diameter_arcsec'] # 1.52 ''
        
    avg_platescale = fiber_dia_asec / fiber_dia # ['' / um]

    exposure_seeing_fwhm *= avg_platescale # [''] 
    
    log.info('Computed effective seeing of {:.6f} arcseconds (focalplane avg.) for a ETC PSF fiberfrac of {:.6f}'.format(exposure_seeing_fwhm, etc_fiberfracs['psf']))
    
    # compute the seeing and plate scale correction                                                                                                                                                                                          
    x_mm = fibermap["FIBER_X"]
    y_mm = fibermap["FIBER_Y"]
    bad = np.isnan(x_mm)|np.isnan(y_mm)
    x_mm[bad]=0.
    y_mm[bad]=0.

    if "DELTA_X" in fibermap.dtype.names :
        dx_mm = fibermap["DELTA_X"] # mm                                                                                                                                                                                                      
    else :
        log.warning("no column 'DELTA_X' in fibermap, assume = zero")
        dx_mm = np.zeros(len(fibermap))

    if "DELTA_Y" in fibermap.dtype.names :
        dy_mm = fibermap["DELTA_Y"] # mm                                                                                                                                                                                                      
    else :
        log.warning("no column 'DELTA_Y' in fibermap, assume = zero")
        dy_mm = np.zeros(len(fibermap))

    bad = np.isnan(dx_mm)|np.isnan(dy_mm)
    dx_mm[bad]=0.
    dy_mm[bad]=0.

    ps = load_platescale()
    isotropic_platescale = np.interp(x_mm**2+y_mm**2,ps['radius']**2,np.sqrt(ps['radial_platescale']*ps['az_platescale'])) # um/arcsec                                                                                                        
    # we could include here a wavelength dependence on seeing, or non-Gaussian.
    avg_sigmas_um  = (exposure_seeing_fwhm/2.35) / avg_platescale # um
    iso_sigmas_um  = (exposure_seeing_fwhm/2.35) * isotropic_platescale # um
    
    offsets_um = np.sqrt(dx_mm**2+dy_mm**2)*1000. # um                                                                                                                                                                                        
    nfibers = len(fibermap)
        
    log.info('Median fiber offset {:.6f} [{:.6f} to {:.6f}]'.format(np.median(offsets_um), offsets_um.min(), offsets_um.max()))
    log.info('Median avg psf microns {:.6f} [{:.6f} to {:.6f}]'.format(np.median(avg_sigmas_um), avg_sigmas_um.min(), avg_sigmas_um.max()))
    log.info('Median iso psf microns {:.6f} [{:.6f} to {:.6f}]'.format(np.median(iso_sigmas_um), iso_sigmas_um.min(), iso_sigmas_um.max()))

    ##
    ##  FIX ME:  desispec flux calibration, in the absence of seeing values in the header, assumes 1.1'' as a default;
    ##
    ##  https://github.com/desihub/desispec/blob/05cd4cdf501b9afb7376bb0ea205517246b58769/py/desispec/scripts/fluxcalibration.py#L56
    ##
    ##  We do the same here until that is corrected:  https://github.com/desihub/desispec/issues/1267
    ##
    iso_sigmas_um_1p1      = (1.1/2.35) * isotropic_platescale

    ##  @Julien: offsets = 0.0?
    psf_like_1p1           = fa.value("POINT",iso_sigmas_um_1p1,offsets=np.zeros_like(iso_sigmas_um_1p1))

    ## 
    tsnr_fiberfracs        = {}

    tsnr_fiberfracs['exposure_seeing_fwhm'] = exposure_seeing_fwhm
    
    if no_offsets:
        offsets_um = np.zeros_like(offsets_um)

        log.info('Zeroing fiber offsets for tsnr fiberfracs.')
    
    # 0.45''
    tsnr_fiberfracs['elg']  = fa.value("DISK", iso_sigmas_um,offsets=offsets_um,hlradii=0.45 * np.ones_like(iso_sigmas_um))    
    tsnr_fiberfracs['elg'] *= etc_fiberfracs['elg'] / fa.value("DISK", avg_sigmas_um,offsets=np.zeros_like(avg_sigmas_um),hlradii=0.45 * np.ones_like(avg_sigmas_um))
    
    # BULGE == DEV.
    tsnr_fiberfracs['bgs']  = fa.value("BULGE",iso_sigmas_um,offsets=offsets_um,hlradii=1.50 * np.ones_like(iso_sigmas_um))
    tsnr_fiberfracs['bgs'] *= etc_fiberfracs['bgs'] / fa.value("BULGE", avg_sigmas_um,offsets=np.zeros_like(avg_sigmas_um),hlradii=1.50 * np.ones_like(avg_sigmas_um))

    # FIX ME: ETC derived LRG FFRAC. BGS IN THE MEANTIME (DOES NOT DETERMINE TILE COMPLETION).
    tsnr_fiberfracs['lrg']  = fa.value("BULGE",iso_sigmas_um,offsets=offsets_um,hlradii=1.00 * np.ones_like(iso_sigmas_um))
    tsnr_fiberfracs['lrg'] *= etc_fiberfracs['bgs'] / fa.value("BULGE", avg_sigmas_um,offsets=np.zeros_like(avg_sigmas_um),hlradii=1.00 * np.ones_like(avg_sigmas_um))

    for tracer in ['mws', 'qso', 'lya']:
        tsnr_fiberfracs[tracer]  = fa.value("POINT",iso_sigmas_um,offsets=offsets_um)
        tsnr_fiberfracs[tracer] *= etc_fiberfracs['psf'] / fa.value("POINT", avg_sigmas_um,offsets=np.zeros_like(avg_sigmas_um))

    for tracer in ['qso', 'elg', 'lrg', 'bgs']:
        log.info("Computed median nominal {} fiber frac of {:.6f} ([{:.6f},{:.6f}]) for a seeing fwhm of: {:.6f} arcseconds.".format(tracer, np.median(tsnr_fiberfracs[tracer]),\
                                                                                                                                                       tsnr_fiberfracs[tracer].min(),\
                                                                                                                                                       tsnr_fiberfracs[tracer].max(),\
                                                                                                                                                       exposure_seeing_fwhm))
    # Normalize.
    tracers = ['mws', 'qso', 'lya', 'elg', 'lrg', 'bgs']

    notnull = psf_like_1p1 > 0.0

    log.info("Correcting fiberfrac for {:d} of {:d} sources.".format(np.count_nonzero(notnull), len(notnull)))
    
    for tracer in tracers:
        # To the TSNR 'signal' we apply the flux calib (including transparency).  Here, we remove the PSF fiberloss accounted for there. 
        tsnr_fiberfracs[tracer][ notnull] /= psf_like_1p1[notnull]
        tsnr_fiberfracs[tracer][~notnull]  = 1.0
        
    return  tsnr_fiberfracs

#- Cache files from desimodel to avoid reading them N>>1 times
_camera_nea_angperpix = None
_band_ensemble = None

def calc_tsnr2(frame, fiberflat, skymodel, fluxcalib, alpha_only=False, model_ivar=True, model_poisson=False, model_extfiberloss=False, components=False, no_offsets=False):
    '''
    Compute template SNR^2 values for a given frame

    Args:
        frame : uncalibrated Frame object for one camera
        fiberflat : FiberFlat object
        sky : SkyModel object
        fluxcalib : FluxCalib object
        model_poisson: if True, add source Poisson term to model ivar (supported for BGS & Backup).

    returns (tsnr2, alpha):
        `tsnr2` dictionary, with keys labeling tracer (bgs,elg,etc.), of values
        holding nfiber length array of the tsnr^2 values for this camera, and
        `alpha`, the relative weighting btwn rdnoise & sky terms to model var.

    Note:  Assumes DESIMODEL is set and up to date.
    '''
    global _camera_nea_angperpix
    global _band_ensemble

    log=get_logger()

    if not (frame.meta["BUNIT"]=="count/Angstrom" or frame.meta["BUNIT"]=="electron/Angstrom" ) :
        log.error("requires an uncalibrated frame")
        raise RuntimeError("requires an uncalibrated frame")

    camera=frame.meta["CAMERA"].strip().lower()
    band=camera[0]

    expid=frame.meta["EXPID"]
    night=frame.meta["NIGHT"]

    etcpath=findfile('etc', night=night, expid=expid)

    if not os.path.exists(etcpath):
        log.critical('Failed to find etc data at {}.  Assuming nominal etc fiberfracs.'.format(etcpath))

        etc_fiberfracs={}
        
        etc_fiberfracs['psf'] = 0.56198
        etc_fiberfracs['elg'] = 0.41220
        etc_fiberfracs['bgs'] = 0.18985
        
    else:
        log.info('Retrieved etc data from {}'.format(etcpath))
        
        with open(etcpath) as f: 
            etcdata = json.load(f)

        etc_fiberfracs={}

        for tracer in ['psf', 'elg', 'bgs']:
            etc_fiberfracs[tracer]=etcdata['expinfo']['ffrac_{}'.format(tracer)]

    tsnr_fiberfracs = calc_tsnr_fiberfracs(frame.fibermap, etc_fiberfracs, no_offsets=no_offsets)

    if not model_extfiberloss:
        # No extended fiberlosses. 
        exposure_seeing_fwhm = tsnr_fiberfracs['exposure_seeing_fwhm']
        
        tsnr_fiberfracs = {}
        tsnr_fiberfracs['exposure_seeing_fwhm'] = exposure_seeing_fwhm
                
    psfpath=findcalibfile([frame.meta],"PSF")
    psf=GaussHermitePSF(psfpath)

    log.info('Retrieved psf data from {}'.format(psfpath))
    
    # Returns bivariate spline to be evaluated at (fiber, wave).
    if not "DESIMODEL" in os.environ :
        msg = "requires $DESIMODEL to get the NEA and the SNR templates"
        log.error(msg)
        raise RuntimeError(msg)

    if _camera_nea_angperpix is None:
        _camera_nea_angperpix = dict()

    if camera in _camera_nea_angperpix:
        nea, angperpix = _camera_nea_angperpix[camera]
    else:
        neafilename=os.path.join(os.environ["DESIMODEL"],
                                 f"data/specpsf/nea/masternea_{camera}.fits")
        log.info("read NEA file {}".format(neafilename))
        nea, angperpix = read_nea(neafilename)
        _camera_nea_angperpix[camera] = nea, angperpix

    if _band_ensemble is None:
        _band_ensemble = dict()

    if band in _band_ensemble:
        ensemble = _band_ensemble[band]
    else:
        ensembledir=os.path.join(os.environ["DESIMODEL"],"data/tsnr")
        log.info("read TSNR ensemble files in {}".format(ensembledir))
        ensemble = get_ensemble(ensembledir, bands=[band,])
        _band_ensemble[band] = ensemble

    nspec, nwave = fluxcalib.calib.shape

    fibers = np.arange(nspec)
    rdnoise = fb_rdnoise(fibers, frame, psf)

    # Extinction. 
    ebv = frame.fibermap['EBV']

    if np.sum(ebv!=0)>0 :
        log.info("{} {} TSNR MEDIAN EBV = {:.3f}".format(frame.meta["EXPID"], camera, np.median(ebv[ebv!=0])))
    else :
        log.info("{} {} TSNR MEDIAN EBV = 0.0".format(frame.meta["EXPID"], camera))
        
    # Evaluate.
    npix = nea(fibers, frame.wave)
    angperpix = angperpix(fibers, frame.wave)
    angperspecbin = np.mean(np.gradient(frame.wave))

    for label, x in zip(['RDNOISE', 'NEA', 'ANGPERPIX', 'ANGPERSPECBIN'], [rdnoise, npix, angperpix, angperspecbin]):
        log.info('{} {} {} \t {:.3f} +- {:.3f}'.format(frame.meta["EXPID"], camera, label.ljust(10), np.median(x), np.std(x)))

    # Relative weighting between rdnoise & sky terms to model var.
    alpha = calc_alpha(frame, fibermap=frame.fibermap,
                rdnoise_sigma=rdnoise, npix_1d=npix,
                angperpix=angperpix, angperspecbin=angperspecbin,
                fiberflat=fiberflat, skymodel=skymodel)

    log.info("{} {} TSNR ALPHA = {:.6f}".format(frame.meta["EXPID"], camera, alpha))

    if alpha_only:
        return {}, alpha

    maskfactor = np.ones_like(frame.mask, dtype=np.float)
    maskfactor[frame.mask > 0] = 0.0
    maskfactor *= (frame.ivar > 0.0)

    tsnrs = {}
    tsnrs_signal = {}
    tsnrs_background = {}
    
    for tracer in ensemble.keys():
        denom = var_model(rdnoise, npix, angperpix, angperspecbin, fiberflat, skymodel, alpha=alpha)

        if model_poisson:
            denom += var_tracer(tracer, frame, angperspecbin, fiberflat, fluxcalib, exposure_seeing_fwhm=tsnr_fiberfracs['exposure_seeing_fwhm'])
        
        wave = ensemble[tracer].wave[band]
        dflux = ensemble[tracer].flux[band]

        if len(frame.wave) != len(wave) or not np.allclose(frame.wave, wave):
            log.warning(f'Resampling {tracer} ensemble wavelength to match input {camera} frame')
            tmp = np.zeros([dflux.shape[0], len(frame.wave)])
            for i in range(dflux.shape[0]):
                tmp[i] = np.interp(frame.wave, wave, dflux[i],
                            left=dflux[i,0], right=dflux[i,-1])
            dflux = tmp
            wave = frame.wave

        # Work in uncalibrated flux units (electrons per angstrom); flux_calib includes exptime. tau.
        # Broadcast.
        dflux = dflux * fluxcalib.calib # [e/A]

        # Wavelength dependent fiber flat;  Multiply or divide - check with Julien.
        result = dflux * fiberflat.fiberflat

        # Apply dust transmission.
        result *= dust_transmission(frame.wave, ebv)
        
        if tracer in tsnr_fiberfracs:
            result *= tsnr_fiberfracs[tracer][:,None]
        else:
            log.warning('Missing {} tracer in tsnr fiberfracs.'.format(tracer))
            
        result = result**2.

        if model_ivar:
            result /= denom

        else:
            result *= frame.ivar
            
        # Eqn. (1) of https://desi.lbl.gov/DocDB/cgi-bin/private/RetrieveFile?docid=4723;filename=sky-monitor-mc-study-v1.pdf;version=2
        tsnrs[tracer] = np.sum(result * maskfactor, axis=1)

        if components:
            tsnrs_signal[tracer] = (fluxcalib.calib * fiberflat.fiberflat * dust_transmission(frame.wave, ebv))**2.

            if tracer in tsnr_fiberfracs:
                tsnrs_signal[tracer] *= tsnr_fiberfracs[tracer][:,None]**2.

            # With the Poisson term, the background is tracer dependent. 
            tsnrs_background[tracer] = denom 

    results=dict()

    for tracer in tsnrs.keys():
        key = 'TSNR2_{}_{}'.format(tracer.upper(), band.upper())
        results[key]=tsnrs[tracer]
        log.info('{} {} {} = {:.6f}'.format(frame.meta["EXPID"], camera, key, np.median(tsnrs[tracer])))

    if components:
        components = {}

        for x, d in zip(['SIGNAL', 'BACKGROUND'], [tsnrs_signal, tsnrs_background]):
            for tracer in d.keys():
                key = 'TSNR2_{}_{}_{}'.format(tracer.upper(), band.upper(), x)
                components[key] = d[tracer]
            
        return results, alpha, tsnr_fiberfracs['exposure_seeing_fwhm'], components
        
    else:
        return results, alpha, tsnr_fiberfracs['exposure_seeing_fwhm']

def tsnr2_to_efftime(tsnr2,target_type,program="DARK",efftime_config=None) :
    """ Converts TSNR2 values to effective exposure time.
    Args:
      tsnr2: TSNR**2 values, float or numpy array
      target_type: str, "ELG","BGS","LYA", or other depending on content of  data/tsnr/tsnr-efftime.yaml
      program: str, "DARK", "BRIGHT", or other depending on content of data/tsnr/tsnr-efftime.yaml
      efftime_config: (optional) dictionary with calibration parameters of the form "TSNR2_{target_type}_TO_EFFTIME_{program}"
    Returns: exptime in seconds, same type and shape if applicable as input tsnr2
    """
    if efftime_config is None :
        efftime_config_filename  = resource_filename('desispec', 'data/tsnr/tsnr-efftime.yaml')
        with open(efftime_config_filename) as f:
            efftime_config = yaml.load(f, Loader=yaml.FullLoader)

    keyword="TSNR2_{}_TO_EFFTIME_{}".format(target_type,program)
    if not keyword in efftime_config :
        message="no calibration for TSNR2_{} to EFFTIME_{}".format(target_type,program)
        raise KeyError(message)
    return efftime_config[keyword]*tsnr2
