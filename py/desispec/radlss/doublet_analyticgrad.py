import time
import numba
import numpy               as     np
import astropy.constants   as     const

from   lines               import lines
from   numba               import jit
from   doublet             import doublet
from   twave               import twave
from   desispec.io.meta    import findfile
from   desispec.resolution import Resolution
from   doublet_postagefit  import doublet_chi2

from   resample_flux       import resample_flux
# from   desispec.interpolation import resample_flux


lightspeed = const.c.to('km/s').value

@jit(nopython=True)
def sig_lambda(z, sigmav, lineb):
    return  sigmav * (1. + z) * lineb / lightspeed

@jit(nopython=True)
def dsigdz(v, lineb):
    return v * lineb / lightspeed

@jit(nopython=True)
def dsigdv(z, lineb):
    return (1. + z) * lineb / lightspeed

@jit(nopython=True)
def dMdlnA(z, v, r, tflux, linea=3726.032, lineb=3728.815):
    return tflux

@jit(nopython=True)
def dMdr(z, v, r, tflux, linea=3726.032, lineb=3728.815):
    # if tflux is None:
    #    _, tflux = doublet(z, twave, sigmav=v, r=r, linea=linea, lineb=lineb)

    sigma_lam = sig_lambda(z=z, sigmav=v, lineb=lineb)

    result    = -tflux / (1. + r) 

    # Only defined for doublets, retain linea. 
    result   += np.exp(-((twave  - linea * (1. + z)) / np.sqrt(2.) / sigma_lam)**2.) / (1. + r) / np.sqrt(2. * np.pi) / sigma_lam / sigma_lam

    # Identically zero if r is zero. 
    return  result

# @jit(nopython=True)
def dMdz(z, v, r, tflux, linea=3726.032, lineb=3728.815):
    sigma_lam = sig_lambda(z=z, sigmav=v, lineb=lineb)
        
    result    = -2. * tflux * dsigdz(v=v, lineb=lineb) / sigma_lam

    def dA1dz(z=z, v=v, r=r, line=lineb):
        def dydz(z=z, v=v, r=r, line=line):
            _dsigdz = dsigdz(v=v, lineb=line)
              
            return - _dsigdz * twave / np.sqrt(2.) / sigma_lam / sigma_lam - (line / np.sqrt(2.) / sigma_lam) * (1. - (1. + z) * _dsigdz / sigma_lam)
            
        y       = (twave - line * (1. + z)) / np.sqrt(2.) / sigma_lam
        interim = -2. * y * np.exp(- y * y) * dydz(z=z, v=v, r=r, line=line)
        
        return interim
        
    result += (1. / (1. + r) / np.sqrt(2. * np.pi) / sigma_lam**2.) * (r * dA1dz(z=z, v=v, r=r, line=linea) + dA1dz(z=z, v=v, r=r, line=lineb))

    return  result

# @jit(nopython=True)
def dMdv(z, v, r, tflux, linea=3726.032, lineb=3728.815):
    sigma_lam = sig_lambda(z=z, sigmav=v, lineb=lineb)
        
    def dA1dv(z, v, r, lineb=lineb, tflux=tflux):
        y   = (twave - lineb * (1. + z)) / np.sqrt(2.) / sigma_lam

        def dydv(z, v, r, lineb=lineb):
            return  - dsigdv(z, lineb) * (twave - lineb * (1. + z)) / np.sqrt(2.) / sigma_lam / sigma_lam

        return  -2. * y * np.exp(-y * y) * dydv(z, v, r, lineb=lineb) 

    result  = -2. * tflux / sigma_lam
    result *=  dsigdv(z, lineb)

    result += ((1. / (1. + r)) / np.sqrt(2. * np.pi) / sigma_lam / sigma_lam) * (r * dA1dv(z, v, r, lineb=linea, tflux=tflux) + dA1dv(z, v, r, lineb=lineb, tflux=tflux))
    
    return result

def doublet_chi2_grad(z, wave, res, flux, ivar, mask, continuum=0.0, v=5., r=0.7, line_flux=None, linea=3726.032, lineb=3728.815):
    _, tflux      = doublet(z=z, twave=twave, sigmav=v, r=r, linea=linea, lineb=lineb)

    model         = resample_flux(wave, twave, tflux)

    # Unit amplitude.
    model         = res.dot(model)

    chi_sq        = ivar * (flux - line_flux * model)**2.
    chi_sq        = np.sum(chi_sq[mask == 0])
        
    # dMdtheta    = [z, v, r, lnA].
    grad          = np.zeros(4)

    for i, fdMdtheta in enumerate([dMdz, dMdv, dMdr, dMdlnA]):
        dMdtheta  = fdMdtheta(z=z, v=v, r=r, tflux=tflux, linea=linea, lineb=lineb)
        dMdtheta  = resample_flux(wave, twave, dMdtheta)
        dMdtheta  = res.dot(dMdtheta)
                
        grad[i]   = -2. * np.sum((flux[mask == 0] - line_flux * model[mask == 0]) * ivar[mask == 0] * line_flux * dMdtheta[mask == 0])
        
    return  grad, chi_sq


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
            # X2   = doublet(z, twave, sigmav=v, r=r, linea=linea, lineb=lineb, index=5514)
            
            result += X2

            break
            
        return  result

    def _fdgrad(x):
        z             = x[0]
        v             = x[1]
        r             = x[2]
        lnA           = x[3]

        line_flux     = np.exp(lnA)

        result        = 0.0

        chisq         = 0.0
        grad          = np.zeros(4)

        eps           = np.sqrt(np.finfo(float).eps)
        
        for cam in cams:
            wave      = postage[cam].wave
            res       = Resolution(postage[cam].R)
            flux      = postage[cam].flux
            ivar      = postage[cam].ivar
            mask      = postage[cam].mask

            for i in np.arange(4):
                hi            = np.array(x, copy=True)
                lo            = np.array(x, copy=True) 

                lo[i]        -= eps
                hi[i]        += eps
                
                _, _, hiX2, _ = doublet_chi2(hi[0], wave, res, flux, ivar, mask, continuum=0.0, sigmav=hi[1], r=hi[2], line_flux=np.exp(hi[3]), linea=linea, lineb=lineb)
                _, _, loX2, _ = doublet_chi2(lo[0], wave, res, flux, ivar, mask, continuum=0.0, sigmav=lo[1], r=lo[2], line_flux=np.exp(lo[3]), linea=linea, lineb=lineb)

                grad[i]      += (hiX2 - loX2) / 2. / eps

            break

        return  grad, -99.
    
    def _agrad(x):
        z             = x[0]
        v             = x[1]
        r             = x[2]
        lnA           = x[3]

        line_flux     = np.exp(lnA)

        result        = 0.0

        chisq         = 0.0
        grad          = np.zeros(4)
        
        for cam in cams:
            wave      = postage[cam].wave
            res       = Resolution(postage[cam].R)
            flux      = postage[cam].flux
            ivar      = postage[cam].ivar
            mask      = postage[cam].mask

            _grad, X2 = doublet_chi2_grad(z, wave, res, flux, ivar, mask, continuum=0.0, v=v, r=r, line_flux=line_flux, linea=linea, lineb=lineb)

            grad     += _grad
            chisq    += X2
            
            break
            
        return  grad, chisq

    ## 
    x0           = np.array([rrz, 52., 0.7, 0.0])        

    ##  Force jit compilation.
    _agrad(x0)
    
    start        = time.time()

    for i in np.arange(30):
        agrad, chisq = _agrad(x0)

    end          = time.time()
        
    print(chisq, _X2(x0))
    print()
    print(agrad)
    print(end - start)

    # print(_fdgrad(x0)[0])

    start       = time.time()
    
    for i in np.arange(30):
        eps     = 1.e-8
        grad    = approx_fprime(x0, _X2, eps)

    end         = time.time()

    print()
    print(grad)
    print(end - start)
    
    # print(dMdz(x0[0], x0[1], x0[2], linea=3726.032, lineb=3728.815, tflux=None)[5514])
    # print(dMdr(x0[0], x0[1], x0[2], linea=3726.032, lineb=3728.815, tflux=None)[5514])

    # print(rrz, 3728.815 * (1. + rrz))
    # print(postage['r5'].wave)
    
    print('\n\nDone.\n\n')
