import astropy.io.fits as fits
import numpy as np

from scipy.interpolate import RectBivariateSpline

def read_nea(path):
    nea=fits.open(path)
    wave=nea['WAVELENGTH'].data
    angperpix=nea['ANGPERPIX'].data
    nea=nea['NEA'].data

    fiber = np.arange(len(nea))

    nea = RectBivariateSpline(fiber, wave, nea)
    angperpix = RectBivariateSpline(fiber, wave, angperpix)
        
    return  nea, angperpix
