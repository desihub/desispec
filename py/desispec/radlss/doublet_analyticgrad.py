import time
import numba
import numpy               as     np
import astropy.constants   as     const

from   lines               import lines
from   numba               import jit
from   doublet             import doublet
from   twave               import twave
from   doublet_postagefit  import doublet_obs
from   desispec.io.meta    import findfile
from   desispec.resolution import Resolution
from   doublet_postagefit  import doublet_chi2
from   resample_flux       import resample_flux


lightspeed = const.c.to('km/s').value

def dsigdz(v, lineb):
    return v * lineb / lightspeed

# Obs. model = Resolution * [exp(\tilde A) * \tilde M].
def dMdlnA(z, v, r, linea=3726.032, lineb=3728.815, tflux=None):
    if tflux is None:
        _, tflux = doublet(z, twave, sigmav=v, r=r, linea=linea, lineb=lineb)

    return tflux

def dMdr(z, v, r, linea=3726.032, lineb=3728.815, tflux=None):
    result    = r / (1. + r)**2.
    result   /= np.sqrt(2. * np.pi)

    sigma_lam = v * (1. + z) * lineb / lightspeed

    result   /= sigma_lam**2.

    # Only defined for doublets, retain linea. 
    result   *= np.exp(-((twave  - linea * (1. + z)) / np.sqrt(2.) / sigma_lam)**2.)

    # Identically zero if r is zero. 
    return result

def dMdz(z, v, r, linea=3726.032, lineb=3728.815, tflux=None):
    if tflux is None:
        _, tflux = doublet(z, twave, sigmav=v, r=r, linea=linea, lineb=lineb)

    sigma_lam = v * (1. + z) * lineb / lightspeed
        
    result  = -2. * tflux / sigma_lam
    result *= dsigdz(v, lineb=lineb) 
    
    def dA1dz(z, v, r, lineb=lineb, tflux=tflux):
        y   = twave - lineb * (1. + z) / np.sqrt(2.) / sigma_lam

        def dydz(z, v, r, lineb=lineb):
            return -1. * dsigdz(v, lineb=lineb) * ((twave - lineb) / np.sqrt(2) / sigma_lam / sigma_lam - z * lineb / np.sqrt(2.) / sigma_lam / sigma_lam)  - lineb / np.sqrt(2.) / sigma_lam
        
        return -2. * y * np.exp(- y * y) * dydz(z, v, r, lineb=lineb) 
        
    result += ((1. / (1. + r)) / np.sqrt(2. * np.pi) / sigma_lam / sigma_lam) * (r * dA1dz(z, v, r, lineb=linea, tflux=tflux) + dA1dz(z, v, r, lineb=lineb, tflux=tflux))

    return  result

def dMdv(z, v, r, linea=3726.032, lineb=3728.815, tflux=None):
    if tflux is None:
        _, tflux = doublet(z, twave, sigmav=v, r=r, linea=linea, lineb=lineb)

    def dA1dv(z, v, r, lineb=lineb, tflux=tflux):
        y   = twave - lineb * (1. + z) / np.sqrt(2.) / v

        def dydv(z, v, r, lineb=lineb):
            return  - dsigdz(v, lineb) * (twave - lineb * (1. + z)) / np.sqrt(2.) / v / v

        return  -2. * y * np.exp(-y * y) * dydv(z, v, r, lineb=lineb) 

    result  = -2. * tflux / v
    result *=  dsigdz(v, lineb=lineb)

    toadd   = ((1. / (1. + r)) / np.sqrt(2. * np.pi) / v / v) * (r * dA1dv(z, v, r, lineb=linea, tflux=tflux) + dA1dv(z, v, r, lineb=lineb, tflux=tflux))
    result += toadd
    
    return result

def doublet_obs(z, wave, res, continuum=0.0, sigmav=5., r=0.1, linea=3726.032, lineb=3728.815):
    _, tflux = doublet(z=z, twave=twave, sigmav=sigmav, r=r, linea=linea, lineb=lineb)
    tflux    = resample_flux(wave, twave, tflux)

    return  res.dot(tflux)

def doublet_chi2_grad(z, wave, res, flux, ivar, mask, continuum=0.0, sigmav=5., r=0.1, line_flux=None, linea=3726.032, lineb=3728.815):
    _, tflux      = doublet(z=z, twave=twave, sigmav=sigmav, r=r, linea=linea, lineb=lineb)

    model         = resample_flux(wave, twave, tflux)

    # Unit amplitude.
    model         = res.dot(model)
    
    # dMdtheta    = [z, v, r, lnA].
    grad          = np.zeros(4)

    for i, fdMdtheta in enumerate([dMdz, dMdv, dMdr, dMdlnA]):
        dMdtheta  = fdMdtheta(z, v, r, linea=linea, lineb=lineb, tflux=tflux)
        dMdtheta  = resample_flux(wave, twave, dMdtheta)
        dMdtheta  = res.dot(dMdtheta)
        dMdtheta *= line_flux
        
        grad[i]   = -2. * np.sum((flux[mask == 0] - line_flux * model[mask == 0]) * ivar[mask == 0] * dMdtheta[mask == 0])
    
    return  grad


if __name__ == '__main__':
    import os
    import pickle
    import pylab as pl
    import astropy.io.fits as fits
    import matplotlib.pyplot as plt

    from   desispec.io import read_frame
    from   astropy.table import Table
    from   scipy.optimize import approx_fprime
    from   cframe_postage import cframe_postage
    from   lines import lines

    
    petal    =  '5'
    fiber    =  11
    night    = '20200315'
    expid    = 55589
    tileid   = '67230'

    ##                                                                                                                                                                                                                                     
    zbest    = Table.read(os.path.join('/global/homes/m/mjwilson/andes/', 'tiles', tileid, night, 'zbest-{}-{}-{}.fits'.format(petal, tileid, night)), 'ZBEST')
    rrz      = zbest['Z'][fiber]
    rrzerr   = zbest['ZERR'][fiber]

    cframes  = {}
    colors   = {'b': 'b', 'r': 'g', 'z': 'r'}

    with open(os.environ['CSCRATCH'] + '/radlss/test/ensemble/template-elg-ensemble-meta.fits', 'rb') as handle:
        meta = pickle.load(handle)['ELG']

    with open(os.environ['CSCRATCH'] + '/radlss/test/ensemble/template-elg-ensemble-flux.fits', 'rb') as handle:
        flux = pickle.load(handle)['ELG']

    with open(os.environ['CSCRATCH'] + '/radlss/test/ensemble/template-elg-ensemble-objmeta.fits', 'rb') as handle:
        objmeta = pickle.load(handle)['ELG']

    for band in ['b', 'r', 'z']:
        cam    = band + petal

        # E.g. ~/andes/exposures/20200315/00055642/cframe-b0-00055642.fits                                                                                                                                                                 
        cframes[cam]               = read_frame(findfile('cframe', night=night, expid=expid, camera=cam, specprod_dir='/global/homes/m/mjwilson/andes/'))
        cframes[cam].flux[fiber,:] = flux[band][115]

    rrz    = meta['REDSHIFT'][115]

    z      = 1.0
    v      = 75.0
    r      = 0.7
    lnA    = 2.5

    _, mod = doublet(z, twave, sigmav=v, r=r)

    postages        = cframe_postage(cframes, fiber, rrz)
    postage         = postages[3]

    # Lower & upper line of OII
    lineida         = 6
    lineidb         = 7 

    postage         = postage[lineidb]
    
    cams            = postage.keys()

    #
    linea           = lines['WAVELENGTH'][lineida]
    lineb           = lines['WAVELENGTH'][lineidb]
    
    def _X2(x):
        z           = x[0]
        v           = x[1]
        r           = x[2]
        lnA         = x[3]

        line_flux   = np.exp(lnA)

        result      = 0.0

        for cam in cams:
            wave = postage[cam].wave
            res  = Resolution(postage[cam].R)
            flux = postage[cam].flux
            ivar = postage[cam].ivar
            mask = postage[cam].mask

            _, _, X2, _ = doublet_chi2(z, wave, res, flux, ivar, mask, continuum=0.0, sigmav=v, r=r, line_flux=line_flux, linea=linea, lineb=lineb)

            result     += X2

            break
            
        return  result

    def _agrad(x):
        z           = x[0]
        v           = x[1]
        r           = x[2]
        lnA         = x[3]

        line_flux   = np.exp(lnA)

        result      = 0.0

        grad        = np.zeros(4)
        
        for cam in cams:
            wave    = postage[cam].wave
            res     = Resolution(postage[cam].R)
            flux    = postage[cam].flux
            ivar    = postage[cam].ivar
            mask    = postage[cam].mask

            grad   += doublet_chi2_grad(z, wave, res, flux, ivar, mask, continuum=0.0, sigmav=v, r=r, line_flux=line_flux, linea=linea, lineb=lineb)

            break
            
        return  grad
        
    x0    = np.array([rrz, 52., 0.7, 6.0])        

    grad  = approx_fprime(x0, _X2, 1.e-8)
    agrad = _agrad(x0)
    
    print(grad)
    print(agrad)
    
    print('\n\nDone.\n\n')
