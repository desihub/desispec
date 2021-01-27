import numpy as np
import astropy.io.fits as fits
import glob
import numpy as np

from desispec.io.spectra import Spectra
from astropy.convolution import convolve, Box1DKernel

def get_ensemble(dirpath, smooth=True):
    paths = glob.glob(dirpath + '/tsnr-ensemble-*.fits')
    bands = ['B', 'R', 'Z']

    wave = {}
    flux = {}
    ivar = {}
    mask = {}
    res = {}

    ensembles = {}

    for path in paths:
        tracer = path.split('/')[-1].split('-')[2]
        dat    = fits.open(path)

        for band in bands:
            wave[band] = dat['WAVE_{}'.format(band)].data
            flux[band] = dat['DFLUX_{}'.format(band)].data
            ivar[band] = 1.e99 * np.ones_like(flux[band])

            if smooth:
                flux[band] = convolve(flux[band][0,:], Box1DKernel(125), boundary='extend')
                flux[band] = flux[band].reshape(1, len(flux[band]))
                
        ensembles[tracer] = Spectra(bands, wave, flux, ivar)

    return  ensembles

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
        
def calc_tsnr(frame, psf, fluxcalib, fiberflat, skymodel, nea, angperpix, ensemble):
    rdnoise = []

    for ifiber in range(len(frame.flux)):
        # quadrants for readnoise.                                                                                                                   
        psf_wave = np.median(frame.wave)
        
        x, y     = psf.xy(ifiber, psf_wave)
        ccd_quad = quadrant(x, y, frame)
        rdnoise.append(frame.meta['OBSRDN{}'.format(ccd_quad)])

    # rdnoise by fiber. 
    rdnoise = np.array(rdnoise)

    '''
    dflux   = np.array(dtemplate_flux, copy=True)

    # Work in uncalibrated flux units (electrons per angstrom); flux_calib includes exptime. tau.
    dflux  *= flux_calib    # [e/A]                                                                                                                                                                

    # Wavelength dependent fiber flat;  Multiply or divide - check with Julien.
    result  = dflux * fiberflat
    result  = result**2.

    # RDNOISE & NPIX assumed wavelength independent.
    denom   = readnoise**2 * npix / angstroms_per_pixel + fiberflat * sky_flux
    result /= denom
    
    # Eqn. (1) of https://desi.lbl.gov/DocDB/cgi-bin/private/RetrieveFile?docid=4723;filename=sky-monitor-mc-study-v1.pdf;version=2
    return  np.sum(result)
    '''
    
    return  None
