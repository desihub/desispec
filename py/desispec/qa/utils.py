""" Module for QA support
"""
from __future__ import print_function, absolute_import, division

import numpy as np

def get_skyres(cframes, sub_sky=False):
    from desispec.io import read_frame
    from desispec.io.sky import read_sky
    from desispec.sky import subtract_sky

    if isinstance(cframes,list):
        all_wave, all_flux, all_res, all_ivar = [], [], [], []
        for cframe_file in cframes:
            wave, flux, res, ivar = get_skyres(cframe_file)
        # Concatenate and return
        all_wave.append(wave)
        all_flux.append(flux)
        all_res.append(res)
        all_ivar.append(ivar)
        return np.concatenate(all_wave), np.concatenate(all_flux), \
               np.concatenate(all_res), np.concatenate(all_ivar)

    cframe = read_frame(cframes)
    if cframe.meta['FLAVOR'] in ['flat','arc']:
        raise ValueError("Bad flavor for exposure: {:s}".format(cframes))

    # Sky
    sky_file = cframes.replace('cframe', 'sky')
    skymodel = read_sky(sky_file)
    if sub_sky:
        subtract_sky(cframe, skymodel)
    # Resid
    skyfibers = np.where(cframe.fibermap['OBJTYPE'] == 'SKY')[0]
    res = cframe.flux[skyfibers]
    ivar = cframe.ivar[skyfibers]
    flux = skymodel.flux[skyfibers]  # Residuals
    wave = np.outer(np.ones(flux.shape[0]), cframe.wave)
    # Return
    return wave.flatten(), flux.flatten(), res.flatten(), ivar.flatten()
