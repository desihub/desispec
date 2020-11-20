import numpy as np

from doublet_obs import doublet_obs


def matchedtemp_lineflux(z, wave, res, flux, ivar, mask, continuum=0.0, sigmav=50.0, r=0.7, linea=3726.032, lineb=3728.815):
    ##  Solve for the best fit line_flux given the observed flux.                                                                                                                                                                         
    ##  Eqn. (12) of https://arxiv.org/pdf/2007.14484.pdf                                                                                                                                                                                 
    rflux         = doublet_obs(z, wave, wave, res, continuum=0.0, sigmav=sigmav, r=r, linea=linea, lineb=lineb)

    mm            = mask == 0
    
    line_flux     = np.sum( flux[mm] * rflux[mm] * ivar[mm])
    line_flux    /= np.sum(rflux[mm] * rflux[mm] * ivar[mm])

    line_flux_err = np.sum(rflux[mm] * rflux[mm] * ivar[mm])
    line_flux_err = 1. / np.sqrt(line_flux_err)

    return  line_flux, line_flux_err
