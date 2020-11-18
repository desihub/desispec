import time
import numba
import numpy as np
import scipy

from   scipy             import optimize
from   scipy.optimize    import approx_fprime, minimize, Bounds
from   scipy.stats       import multivariate_normal
from   jax               import grad, jit, vmap, hessian, jacfwd, jacrev
from   jax.experimental  import optimizers
from   scipy.optimize    import leastsq
from   numba             import jit

# from   desispec.interpolation import resample_flux
from   resample_flux import resample_flux

from   doublet import doublet
from   desispec.io import read_frame
from   desispec.io.meta import findfile
from   desispec.resolution import Resolution
from   doublet_priors import mlogprior
from   cframe_postage import cframe_postage
from   lines import lines, ugroups
from   twave import twave


def doublet_obs(z, twave, wave, res, continuum=0.0, sigmav=5., r=0.1, linea=3726.032, lineb=3728.815):
    _, tflux = doublet(z=z, twave=twave, sigmav=sigmav, r=r, linea=linea, lineb=lineb)
    tflux    = resample_flux(wave, twave, tflux)

    return  res.dot(tflux)

def doublet_chi2(z, twave, wave, res, flux, ivar, mask, continuum=0.0, sigmav=5., r=0.1, line_flux=None, linea=3726.032, lineb=3728.815):    
    '''
    Given a redshift, cframe (extracted wave, res, flux, ivar) return 
    chi sq. for a doublet line model of given parameters, e.g. line flux.

    If line flux is None, estimate it first. 
    '''
    
    rflux             = doublet_obs(z, twave, wave, res, continuum=0.0, sigmav=sigmav, r=r, linea=linea, lineb=lineb)
    '''
    if line_flux is None:
        ##  Solve for the best fit line_flux given the observed flux.
        ##  Eqn. (12) of https://arxiv.org/pdf/2007.14484.pdf
        line_flux     = np.sum( flux * rflux * ivar)
        line_flux    /= np.sum(rflux * rflux * ivar)
    
        line_flux_err = np.sum(rflux * rflux * ivar)
        line_flux_err = 1. / np.sqrt(line_flux_err)
     ''' 
    rflux            *= line_flux
        
    X2                = ivar * (flux - rflux)**2.

    result            = np.sum(X2[mask == 0])
    
    return  wave, rflux, result, line_flux

def doublet_fit(x0, rrz, rrzerr, postages, lineida=6, lineidb=7, plot=False):
    '''
    Passed a postage stamp, find the best fit Gaussian (doublet).
    '''

    start           = time.time()
    
    postage         = postages[lineidb]
    cams            = list(postage.keys())

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
            wave    = postage[cam].wave
            res     = Resolution(postage[cam].R)
            flux    = postage[cam].flux
            ivar    = postage[cam].ivar
            mask    = postage[cam].mask

            tw      = twave(wave.min(), wave.max())

            _, _, X2, _ = doublet_chi2(z, tw, wave, res, flux, ivar, mask, continuum=0.0, sigmav=v, r=r, line_flux=line_flux, linea=linea, lineb=lineb)

            result += X2
            
        return  result

    def _res(x):
        z           = x[0]
        v           = x[1]
        r           = x[2]
        lnA         = x[3]

        line_flux   = np.exp(lnA)

        residuals   = []

        for cam in cams:
            wave    = postage[cam].wave
            res     = Resolution(postage[cam].R)
            flux    = postage[cam].flux
            ivar    = postage[cam].ivar
            mask    = postage[cam].mask == 0

            tw      = twave(wave.min(), wave.max())
            
            rflux   = doublet_obs(z, tw, wave, res, continuum=0.0, sigmav=v, r=r, linea=linea, lineb=lineb)
            rflux  *= line_flux

            res     = np.sqrt(ivar[mask]) * (flux[mask] - rflux[mask])

            residuals += res.tolist()

        residuals      = np.array(residuals) 
            
        return  residuals
        
    def mloglike(x):
        return  _X2(x) / 2.

    def mlogpos(x):
        return  mloglike(x) # + mlogprior(x, rrz, rrzerr)

    def scipy_gradient(x):
        eps = 1.e-8
        
        return optimize.approx_fprime(x, mlogpos, eps)

    '''
    _jax_gradient_mlogpos =  grad(mlogpos)
    _jax_hessian_mlogpos  = jacfwd(jacrev(mlogpos))

    def hvp(f, x, v):
        return grad(lambda x: jnp.vdot(grad(f)(x), v))(x)
    
    def jax_gradient(x):
        return  _jax_gradient_mlogpos(x)
    
    def jax_hessian(x):
        return _jax_hessian_mlogpos(x)

    def jax_hessian_vec_prod(x, p):
        return  grad(lambda x: np.vdot(grad(mlogpos)(x), p))(x)
    
    
    opt_init, opt_update, get_params = optimizers.momentum(step_size=1e-3, mass=0.9)

    # @jit
    def step(i, opt_state):
        params = get_params(opt_state)
        g      = jax_gradient(params)
        
        return  opt_update(i, g, opt_state)

    opt_state      = opt_init(x0)

    for i in range(10):
        opt_state  = step(i, opt_state)
        net_params = get_params(opt_state)

    print(net_params)
    '''
    # print('{} \t {:.6e} \t {:.6e}'.format(gradient(x0), mlogpos(x0), _X2(x0)))
    '''
    # Parameters which minimize f, i.e., f(xopt) == fopt.
    # Minimum value.
    # Value of gradient at minimum, fprime(xopt), which should be near 0.
    # Value of 1/fdoubleprime(xopt), i.e., the inverse Hessian matrix.
    # Number of function_calls made.
    # Number of gradient calls made.
    # Maximum number of iterations exceeded. 2 : Gradient and/or function calls not changing. 3 : NaN result encountered. 
    # The value of xopt at each iteration. Only returned if retall is True.
    #
    # See:  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_bfgs.html
    
    result         = scipy.optimize.fmin_bfgs(mlogpos, x0, fprime=jax_gradient, gtol=1e-05, norm=-np.inf, epsilon=1.4901161193847656e-08, maxiter=None, full_output=True, disp=1, retall=1, callback=None)    
    [z, v, r, lnA] = result[0]

    ihess          = result[3]

    # https://astrostatistics.psu.edu/su11scma5/HeavensLecturesSCMAVfull.pdf
    # Note:  marginal errors.
    merr           = np.sqrt(np.diag(ihess))
    '''

    # 'L-BFGS-B'; hess=jax_hessian; hessp=jax_hessian_vec_prod; 'maxiter': 10, 'maxfev': 50; 'xtol': 1.e-0; 'gtol': 1e-6
    # methods        = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'trust-ncg']
    # result         = scipy.optimize.minimize(mlogpos, x0, method=methods[3], jac=scipy_gradient, options={'disp': True, 'return_all': True, 'ftol': 0.25})
    # [z, v, r, lnA] = result.x
    # print([z, v, r, lnA])

    # method='lm'
    result         = scipy.optimize.least_squares(_res, x0, verbose=1, ftol=0.25, max_nfev=6)
    [z, v, r, lnA] = result.x

    print(time.time() - start)
    
    ##
    rflux          = {}
    
    for cam in cams:
        wave        = postage[cam].wave
        res         = Resolution(postage[cam].R)

        tw          = twave(wave.min(), wave.max())
        
        rflux[cam]  = doublet_obs(z, tw, wave, res, continuum=0.0, sigmav=v, r=r, linea=linea, lineb=lineb)
        rflux[cam] *= np.exp(lnA)
        
    if plot:
        import pylab as pl

        for cam in cams:
            wave = postage[cam].wave
            flux = postage[cam].flux
            mask = postage[cam].mask
            
            pl.plot(wave[mask == 0],  flux[mask == 0], label='Observed {}'.format(cam), alpha=0.5)
            pl.plot(wave[mask == 0], rflux[cam][mask == 0], label='Model {}'.format(cam))
            
            pl.xlabel('Wavlength [Angstroms]')
            pl.ylabel('Flux')

            pl.legend(frameon=False)
            
            # pl.xlim(xmin, xmax)
            pl.ylim(bottom=-0.5)
        
        pl.show()
    
    return  result, rflux

if __name__ == '__main__':
    import os
    import pickle
    import pylab as pl
    import astropy.io.fits as fits
    import matplotlib.pyplot as plt
    
    from   astropy.table import Table

    
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
        
    rrz = meta['REDSHIFT'][115]

    start = time.time()
    
    # for i in np.arange(5000):
    # [Group][Line Index][Camera].
    postages = cframe_postage(cframes, fiber, rrz)

    print(time.time() - start)
        
    ncol = 23
    '''
    ##  Plot.                                                                                                                                                                                                                              
    fig, axes = plt.subplots(len(ugroups), ncol, figsize=(50,10))

    groups    = list(postages.keys())
    
    for u in groups:
        index = 0

        for i, x in enumerate(postages[u]):
            for cam in postages[u][x]:
                axes[u,index].axvline((1. + rrz) * lines['WAVELENGTH'][x], c='k', lw=0.5)
                axes[u,index].plot(postages[u][x][cam].wave, postages[u][x][cam].flux, alpha=0.5, lw=0.75, c=colors[cam[0]])
                axes[u,index].set_title(lines['NAME'][x])

                index += 1
                
    fig.suptitle('Fiber {} of petal {} with redshift {:2f}'.format(fiber, petal, rrz), y=1.02)

    plt.tight_layout()

    pl.savefig('plots/postages.pdf')

    pl.clf()
    '''

    x0            = np.array([rrz, 100., 0.7, 6.0])
    
    # Force compilation.
    doublet_fit(x0, rrz, rrzerr, postages[3], lineida=6, lineidb=7, plot=False)
        
    # Starting guess: [z, v, r, lnA]
    x0            = np.array([rrz, 52., 0.7, 6.0])

    result, rflux = doublet_fit(x0, rrz, rrzerr, postages[3], lineida=6, lineidb=7, plot=True)
    
    print('\n\nDone.\n\n')
