import numpy as np
import astropy.io.fits as fits
import glob
import numpy as np

from desispec.io.spectra import Spectra
from astropy.convolution import convolve, Box1DKernel
from scipy.interpolate import RectBivariateSpline
from specter.psf.gausshermite  import  GaussHermitePSF
from desispec.specscore import append_frame_scores
from desiutil.log import get_logger
from scipy.optimize import minimize

def get_ensemble(dirpath, bands, smooth=True):
    paths = glob.glob(dirpath + '/tsnr-ensemble-*.fits')

    wave = {}
    flux = {}
    ivar = {}
    mask = {}
    res = {}

    ensembles = {}

    for path in paths:
        tracer = path.split('/')[-1].split('-')[2].replace('.fits','')
        dat    = fits.open(path)

        for band in bands:
            wave[band] = dat['WAVE_{}'.format(band.upper())].data
            flux[band] = dat['DFLUX_{}'.format(band.upper())].data
            ivar[band] = 1.e99 * np.ones_like(flux[band])

            if smooth:
                flux[band] = convolve(flux[band][0,:], Box1DKernel(125), boundary='extend')
                flux[band] = flux[band].reshape(1, len(flux[band]))

        ensembles[tracer] = Spectra(bands, wave, flux, ivar)

    return  ensembles

def read_nea(path):
    nea=fits.open(path)
    wave=nea['WAVELENGTH'].data
    angperpix=nea['ANGPERPIX'].data
    nea=nea['NEA'].data

    fiber = np.arange(len(nea))

    nea = RectBivariateSpline(fiber, wave, nea)
    angperpix = RectBivariateSpline(fiber, wave, angperpix)

    return  nea, angperpix

def fb_rdnoise(fibers, frame, psf):
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
        return (alpha * rdnoise_variance, fiberflat.fiberflat * skymodel.flux)

    else:
        return alpha * rdnoise_variance + fiberflat.fiberflat * skymodel.flux

def calc_alpha(frame, fibermap, rdnoise_sigma, npix_1d, angperpix, angperspecbin, fiberflat, skymodel):
    '''
    Model Var = alpha * rdnoise component + sky.

    Calcualte the best-fit alpha using the sky fibers
    available to the frame.
    '''

    sky_indx = np.where(fibermap['OBJTYPE'] == 'SKY')[0]
    rd_var, sky_var = var_model(rdnoise_sigma, npix_1d, angperpix, angperspecbin, fiberflat, skymodel, alpha=1.0, components=True)

    def calc_alphavar(alpha):
        return alpha * rd_var[sky_indx,:] + sky_var[sky_indx,:]

    def alpha_X2(alpha):
        _var = calc_alphavar(alpha)
        _ivar =  1. / _var
        X2 = (frame.ivar[sky_indx,:] - _ivar)**2.
        return np.sum(X2)

    res = minimize(alpha_X2, x0=[1.])
    alpha = res.x[0]

    return alpha

def calc_tsnr(bands, neadir, ensembledir, psfpath, frame, uframe, fluxcalib, fiberflat, skymodel, fibermap):
    log=get_logger()

    psf=GaussHermitePSF(psfpath)

    # Returns bivariate splie to be evaluated at (fiber, wave).
    nea, angperpix=read_nea(neadir)
    ensemble=get_ensemble(ensembledir, bands=bands)

    nspec, nwave = fluxcalib.calib.shape

    fibers = np.arange(nspec)
    rdnoise = fb_rdnoise(fibers, uframe, psf)

    # Evaluate.
    npix = nea(fibers, uframe.wave)
    angperpix = angperpix(fibers, uframe.wave)
    angperspecbin = np.mean(np.gradient(uframe.wave))

    for label, x in zip(['RDNOISE', 'NEA', 'ANGPERPIX', 'ANGPERSPECBIN'], [rdnoise, npix, angperpix, angperspecbin]):
        log.info('{} \t {:.3f} +- {:.3f}'.format(label.ljust(10), np.median(x), np.std(x)))

    # Relative weighting between rdnoise & sky terms to model var.
    if fibermap is not None:
        # alpha calc. introduces calibration-dependent (c)frame.ivar dependence. Use uncalibrated.
        alpha = calc_alpha(uframe, fibermap, rdnoise, npix, angperpix, angperspecbin, fiberflat, skymodel)
    else:
        alpha = 1.0

    log.info('TSNR ALPHA = {:.6f}'.format(alpha))

    tsnrs = {}

    for tracer in ensemble.keys():
        tsnrs[tracer] = {}

        for band in bands:
            wave = ensemble[tracer].wave[band]
            dflux = ensemble[tracer].flux[band]

            np.allclose(uframe.wave, wave)

            # Work in uncalibrated flux units (electrons per angstrom); flux_calib includes exptime. tau.
            # Broadcast.
            dflux = dflux * fluxcalib.calib # [e/A]

            # Wavelength dependent fiber flat;  Multiply or divide - check with Julien.
            result = dflux * fiberflat.fiberflat
            result = result**2.

            denom   = var_model(rdnoise, npix, angperpix, angperspecbin, fiberflat, skymodel, alpha=alpha)
            result /= denom

            # Eqn. (1) of https://desi.lbl.gov/DocDB/cgi-bin/private/RetrieveFile?docid=4723;filename=sky-monitor-mc-study-v1.pdf;version=2
            tsnrs[tracer][band] = np.sum(result, axis=1)

    for tracer in tsnrs.keys():
        key = tracer.upper() + 'TSNR_{}'.format(band.upper())
        score = {key: tsnrs[tracer][band]}
        comments={key: ''}

        append_frame_scores(frame,score,comments,overwrite=True)

        log.info('TSNR {} = {:.6f}'.format(key, np.median(tsnrs[tracer][band])))

    frame.meta['TSNRALP'] = alpha
