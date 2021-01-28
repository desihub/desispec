import numpy as np
import astropy.io.fits as fits
import glob
import numpy as np

from desispec.io.spectra import Spectra
from astropy.convolution import convolve, Box1DKernel
from scipy.interpolate import RectBivariateSpline
from specter.psf.gausshermite  import  GaussHermitePSF
from desispec.specscore import append_frame_scores

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

def quadrant(x, y, frame):
    ccdsizes = np.array(frame.meta['CCDSIZE'].split(',')).astype(np.int)

    if (x < (ccdsizes[0] / 2)):
        if (y < (ccdsizes[1] / 2)):
            return  'A'
        else:
            return  'C'

    else:
        if (y < (ccdsizes[1] / 2)):
            return  'B'
        else:
            return  'D'
        
def calc_tsnr(bands, neadir, ensembledir, psfpath, frame, fluxcalib, fiberflat, skymodel):
    psf=GaussHermitePSF(psfpath)
    
    nea, angperpix=read_nea(neadir)
    ensemble=get_ensemble(ensembledir, bands=bands)

    nspec, nwave = fluxcalib.calib.shape
    
    fibers = np.arange(nspec)
    rdnoise = []

    for ifiber in fibers:
        # quadrants for readnoise.                                                                                                                   
        psf_wave = np.median(frame.wave)
        
        x, y     = psf.xy(ifiber, psf_wave)
        ccd_quad = quadrant(x, y, frame)
        rdnoise.append(frame.meta['OBSRDN{}'.format(ccd_quad)])

    # rdnoise by fiber. 
    rdnoise = np.array(rdnoise)    

    tsnrs = {}

    npix = nea(fibers, frame.wave)
    angperpix = angperpix(fibers, frame.wave)

    for tracer in ensemble.keys():
        tsnrs[tracer] = {}
        
        for band in bands:
            wave = ensemble[tracer].wave[band]
            dflux = ensemble[tracer].flux[band]

            np.allclose(frame.wave, wave)
            
            # Work in uncalibrated flux units (electrons per angstrom); flux_calib includes exptime. tau.
            # Broadcast.
            dflux = dflux * fluxcalib.calib # [e/A]
            
            # Wavelength dependent fiber flat;  Multiply or divide - check with Julien.
            result = dflux * fiberflat.fiberflat
            result = result**2.
            
            # RDNOISE & NPIX assumed wavelength independent.
            denom   = rdnoise[:,None]**2 * npix / angperpix + fiberflat.fiberflat * skymodel.flux
            result /= denom
            
            # Eqn. (1) of https://desi.lbl.gov/DocDB/cgi-bin/private/RetrieveFile?docid=4723;filename=sky-monitor-mc-study-v1.pdf;version=2
            tsnrs[tracer][band] = np.sum(result, axis=1)

    for tracer in tsnrs.keys():
        key = tracer.upper() + 'TSNR_{}'.format(band.upper())
        score = {key: tsnrs[tracer][band]}
        comments={key: ''}

        append_frame_scores(frame,score,comments,overwrite=True)
