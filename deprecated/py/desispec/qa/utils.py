"""
desispec.qa.utils
=================

Module for QA support.
"""
from __future__ import print_function, absolute_import, division

import numpy as np

def get_skyres(cframes, sub_sky=False, flatten=True):
    """
    Args:
        cframes: str or list
          Single cframe or a list of them
        sub_sky: bool, optional
          Subtract the sky?  This should probably not be done
        flatten: bool, optional
          Return a flat, 1D array for each variable
        combine: bool, optional
          combine the individual sky fibers?  Median 'smash'

    Returns:
        wave : ndarray
        flux : ndarray
        res : ndarray
        ivar : ndarray

    """
    from desispec.io import read_frame
    from desispec.io.sky import read_sky
    from desispec.sky import subtract_sky

    if isinstance(cframes,list):
        all_wave, all_flux, all_res, all_ivar = [], [], [], []
        for cframe_file in cframes:
            wave, flux, res, ivar = get_skyres(cframe_file, flatten=flatten)
            # Save
            all_wave.append(wave)
            all_flux.append(flux)
            all_res.append(res)
            all_ivar.append(ivar)
        # Concatenate -- Shape is preserved (nfibers, npix)
        twave = np.concatenate(all_wave)
        tflux = np.concatenate(all_flux)
        tres = np.concatenate(all_res)
        tivar = np.concatenate(all_ivar)
        # Return
        return twave, tflux, tres, tivar

    cframe = read_frame(cframes, skip_resolution=True)
    if cframe.meta['FLAVOR'] in ['flat','arc']:
        raise ValueError("Bad flavor for exposure: {:s}".format(cframes))

    # Sky
    sky_file = cframes.replace('cframe', 'sky')
    skymodel = read_sky(sky_file)
    if sub_sky:
        subtract_sky(cframe, skymodel)
    # Resid
    skyfibers = np.where(cframe.fibermap['OBJTYPE'] == 'SKY')[0]
    res = cframe.flux[skyfibers]  # Flux calibrated
    ivar = cframe.ivar[skyfibers] # Flux calibrated
    flux = skymodel.flux[skyfibers]  # Residuals; not flux calibrated!
    wave = np.outer(np.ones(flux.shape[0]), cframe.wave)
    # Combine?
    '''
    if combine:
        res = np.median(res, axis=0)
        ivar = np.median(ivar, axis=0)
        flux = np.median(flux, axis=0)
        wave = np.median(wave, axis=0)
    '''
    # Return
    if flatten:
        return wave.flatten(), flux.flatten(), res.flatten(), ivar.flatten()
    else:
        return wave, flux, res, ivar
