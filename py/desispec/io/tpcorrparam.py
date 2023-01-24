"""
desispec.io.tpcorrparam
=======================

Please add module-level documentation.
"""
from astropy.io import fits
import desispec.tpcorrparam


def read_tpcorrparam(fn):
    mean = fits.getdata(fn, 'MEAN')
    spatial = fits.getdata(fn, 'SPATIAL')
    pca = fits.getdata(fn, 'PCA')
    return desispec.tpcorrparam.TPCorrParam(mean, spatial, pca)
