import numpy as np

from doublet import doublet


# from   desispec.interpolation import resample_flux                                                                                                                                                                                      
from   resample_flux import resample_flux


def doublet_obs(z, twave, wave, res, continuum=0.0, sigmav=5., r=0.1, linea=3726.032, lineb=3728.815):
    _, tflux = doublet(z=z, twave=twave, sigmav=sigmav, r=r, linea=linea, lineb=lineb)

    # print(z, twave.min(), twave.max(), tflux.max(), sigmav, r, linea, lineb)
    
    tflux = resample_flux(wave, twave, tflux)

    return  res.dot(tflux)
