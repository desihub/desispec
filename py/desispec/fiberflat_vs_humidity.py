"""
desispec.fiberflat_vs_humidity
==================

Utility functions to compute a fiber flat corrected for variations with humidity in the shack
"""
from __future__ import absolute_import, division


import numpy as np


def _interpolated_fiberflat_vs_humidity(fiberflat_vs_humidity , humidity_array, shiftcomp, shiftdelta, humidity_point) :

    i1=np.where(humidity_array<humidity_point)[0][-1]
    i2=i1+1
    if i2>=humidity_array.size : # return largest value
        return fiberflat_vs_humidity[-1]

    w1=(humidity_array[i2]-humidity_point)/(humidity_array[i2]-humidity_array[i1])
    w2=(humidity_point-humidity_array[i1])/(humidity_array[i2]-humidity_array[i1])

    if shiftcomp is None or shiftdelta is None : # simple version: pure linear interpolation
        return w1*fiberflat_vs_humidity[i1]+w2*fiberflat_vs_humidity[i2]

    # more complex: mix of wavelength shift and linear interpolation of the rest
    nhum=fiberflat_vs_humidity.shape[0]
    nfiber=fiberflat_vs_humidity.shape[1]
    nwave=fiberflat_vs_humidity.shape[2]

    u=np.arange(nwave)
    flat1=np.zeros((nfiber,nwave))
    flat=np.zeros((nfiber,nwave))
    for fiber in range(nfiber) :
        dflat1=fiberflat_vs_humidity[i1,fiber]-np.interp(u,u+shiftdelta[fiber,i1],shiftcomp[fiber])
        dflat2=fiberflat_vs_humidity[i2,fiber]-np.interp(u,u+shiftdelta[fiber,i2],shiftcomp[fiber])
        shift_point=w1*shiftdelta[fiber,i1]+w2*shiftdelta[fiber,i2]
        flat[fiber]=w1*dflat1+w2*dflat2+np.interp(u,u+shift_point,shiftcomp[fiber])
    return flat

def compute_humidity_corrected_fiberflat(calib_fiberflat, mean_fiberflat_vs_humidity , humidity_array, shiftcomp, shiftdelta, calib_humidity, current_humidity) :

    # interpolate flat for humidity during calibration exposures and for the current value
    mean_fiberflat_at_current_humidity = _interpolated_fiberflat_vs_humidity(mean_fiberflat_vs_humidity , humidity_array, shiftcomp, shiftdelta, current_humidity)
    mean_fiberflat_at_calib_humidity = _interpolated_fiberflat_vs_humidity(mean_fiberflat_vs_humidity , humidity_array, shiftcomp, shiftdelta, calib_humidity)

    # apply humidity correction to current calib fiberflat
    current_fiberflat = calib_fiberflat # do we want to make a copy?
    current_fiberflat.fiberflat *= mean_fiberflat_at_current_humidity/mean_fiberflat_at_calib_humidity
    return current_fiberflat
