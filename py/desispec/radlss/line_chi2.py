import numpy as np

from  desispec.interpolation import resample_flux
from  doublet import doublet
from  desispec.io import read_frame
from  desispec.io.meta import findfile
from  desispec.resolution import Resolution


def line_chi2(z, wave, res, flux, ivar, continuum=0.0, sigmav=5., r=0.1, line_flux=None, lineida=6, lineidb=7, _twave=None):    
    '''
    Given a redshift, cframe (extracted wave, res, flux, ivar) return 
    chi sq. for a doublet line model of given parameters, e.g. line flux.

    If line flux is None, estimate it first. 
    '''

    if _twave is None:
        _twave      = np.arange(3100., 10400., 0.1)
        
    _, _tflux       = doublet(z=z, sigmav=sigmav, r=r, lineida=lineida, lineidb=lineidb, _twave=_twave)
    tflux           = resample_flux(wave, _twave, _tflux)
    
    rflux           = res.dot(tflux)
    
    if line_flux is None:
        ##  Solve for the best fit line_flux given the observed flux.
        ##  Eqn. (12) of https://arxiv.org/pdf/2007.14484.pdf
        line_flux     = np.sum( flux * rflux * ivar)
        line_flux    /= np.sum(rflux * rflux * ivar)
    
        line_flux_err = np.sum(rflux * rflux * ivar)
        line_flux_err = 1. / np.sqrt(line_flux_err)

    rflux          *= line_flux
    rflux          += continuum

    X2              = np.sum((flux - rflux)**2. * ivar)
    
    return  wave, rflux, X2, line_flux


if __name__ == '__main__':
    # E.g. ~/andes/exposures/20200315/00055642/cframe-b0-00055642.fits
    cframe = read_frame(findfile('cframe', night=20200315, expid=55642, camera='b0', specprod_dir='/global/homes/m/mjwilson/andes/'))

    redshift = 1.00
    fiber = 10
    
    wave = cframe.wave
    res = Resolution(cframe.resolution_data[fiber])
    flux = cframe.flux[fiber, :]
    ivar = cframe.ivar[fiber,:]
    
    wave, Rflux, X2, line_flux = line_chi2(redshift, wave, res, flux, ivar, line_flux=1.)

    print(X2)
    
    print('\n\nDone.\n\n')
