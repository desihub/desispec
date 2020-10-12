import os

import glob
import fitsio

import itertools
import warnings

import numpy as np
import pylab as pl

import desisim.templates
import astropy.io.fits           as      fits

import desispec.io
import redrock.templates
import matplotlib.pyplot         as      plt

from   numba                     import  jit
from   astropy.convolution       import  convolve, Box1DKernel
from   desispec.spectra          import  Spectra
from   desispec.frame            import  Frame
from   desispec.resolution       import  Resolution
from   desispec.io.meta          import  findfile
from   desispec.io               import  read_frame, read_fiberflat, read_flux_calibration, read_sky, read_fibermap 
from   desispec.interpolation    import  resample_flux
from   astropy.table             import  Table
from   desispec.io.image         import  read_image
from   specter.psf.gausshermite  import  GaussHermitePSF
from   scipy.signal              import  medfilt
from   desispec.calibfinder      import  CalibFinder
from   astropy.utils.exceptions  import  AstropyWarning
from   scipy                     import  stats
from   pathlib                   import  Path

@jit
def templateSNR(dtemplate_flux, sky_flux=None, flux_calib=None, fiberflat=None, readnoise=None, npix=None, angstroms_per_pixel=None, fiberloss=None, flux_ivar=None):
    """
    Calculate template SNR, given either a model IVAR or cframe ivar.
        
    Args:
        template_flux: Original - 100A smoothed spectrum [ergs/s/cm2/A]. 
        sky_flux: electrons/A
        flux_calib: 
        readnoise:electrons/pixel
        npix:
        angstroms_per_pixel: in the wavelength direction.
        fiberloss:
        flux_ivar: Equivalent to calibrated flux [ergs/s/cm2/A]. 
            
    Optional inputs:
        ...

    Returns:
        template S/N per camera.
    """

    dflux = dtemplate_flux
    
    if flux_ivar is not None:
        # Work in calibrated flux units.
        # Assumes Poisson Variance from source is negligible.
        return  np.sum(flux_ivar * dflux ** 2.)

    else: 
        # Work in uncalibrated flux units (electrons per angstrom); flux_calib includes exptime. tau.
        dflux  *= flux_calib    # [e/A]
 
        # Wavelength dependent fiber flat;  Multiply or divide - check with Julien. 
        result  = dflux * fiberflat
        result  = result**2.
    
        if fiberloss is not None:
          lossless_fluxcalib = flux_calib / fiberloss
    
        # RDNOISE & NPIX assumed wavelength independent.
        denom   = readnoise**2 * npix / angstroms_per_pixel + fiberflat * sky_flux
        
        result /= denom
        
        # Eqn. (1) of https://desi.lbl.gov/DocDB/cgi-bin/private/RetrieveFile?docid=4723;filename=sky-monitor-mc-study-v1.pdf;version=2
        return  np.sum(result)
