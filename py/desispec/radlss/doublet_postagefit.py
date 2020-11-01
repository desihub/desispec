import numpy as np
import scipy

from   scipy             import optimize
from   scipy.optimize    import approx_fprime, minimize, Bounds
from   scipy.stats       import multivariate_normal

from   desispec.interpolation import resample_flux
from   doublet import doublet
from   desispec.io import read_frame
from   desispec.io.meta import findfile
from   desispec.resolution import Resolution
from   doublet_priors import mlogprior
from   cframe_postage import cframe_postage
from   lines import lines, ugroups


def doublet_obs(z, wave, res, continuum=0.0, sigmav=5., r=0.1, lineida=6, lineidb=7, _twave=None):
    if _twave is None:
        _twave = np.arange(3100., 10400., 0.1)

    _, _tflux  = doublet(z=z, sigmav=sigmav, r=r, lineida=lineida, lineidb=lineidb, _twave=_twave)
    tflux      = resample_flux(wave, _twave, _tflux)

    return  res.dot(tflux)
    
def doublet_chi2(z, wave, res, flux, ivar, mask, continuum=0.0, sigmav=5., r=0.1, line_flux=None, lineida=6, lineidb=7, _twave=None):    
    '''
    Given a redshift, cframe (extracted wave, res, flux, ivar) return 
    chi sq. for a doublet line model of given parameters, e.g. line flux.

    If line flux is None, estimate it first. 
    '''

    if _twave is None:
        _twave      = np.arange(3100., 10400., 0.1)
            
    rflux           = doublet_obs(z, wave, res, continuum=0.0, sigmav=sigmav, r=r, lineida=6, lineidb=7, _twave=None)
    
    if line_flux is None:
        ##  Solve for the best fit line_flux given the observed flux.
        ##  Eqn. (12) of https://arxiv.org/pdf/2007.14484.pdf
        line_flux     = np.sum( flux * rflux * ivar)
        line_flux    /= np.sum(rflux * rflux * ivar)
    
        line_flux_err = np.sum(rflux * rflux * ivar)
        line_flux_err = 1. / np.sqrt(line_flux_err)

    rflux          *= line_flux
    # rflux        += continuum

    X2              = (flux - rflux)**2. * ivar

    result          = np.sum(X2[mask == 0])
    
    return  wave, rflux, result, line_flux

def doublet_fit(x0, rrz, rrzerr, postages, lineida=None, lineidb=None, plot=False):
    '''
    Passed a postage stamp, find the best fit Gaussian (doublet).
    '''
    
    postage         = postages[lineidb]
    cams            = list(postage.keys())
    
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
            
            _, _, X2, _ = doublet_chi2(z, wave, res, flux, ivar, mask, continuum=0.0, sigmav=v, r=r, line_flux=line_flux, lineida=lineida, lineidb=lineidb)

            result += X2
            
        return  result

    def mloglike(x):
        return  _X2(x) / 2.

    def mlogpos(x):
        return  mloglike(x) # + mlogprior(x, rrz, rrzerr)

    def gradient(x):
        eps = 1.e-8
        
        return optimize.approx_fprime(x, mlogpos, eps)

    # print('{} \t {:.6e} \t {:.6e}'.format(gradient(x0), mlogpos(x0), _X2(x0)))
    
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
    result         = scipy.optimize.fmin_bfgs(mlogpos, x0, fprime=gradient, gtol=1e-05, norm=-np.inf, epsilon=1.4901161193847656e-08, maxiter=None, full_output=True, disp=1, retall=1, callback=None)    
    [z, v, r, lnA] = result[0]

    ihess          = result[3]

    # https://astrostatistics.psu.edu/su11scma5/HeavensLecturesSCMAVfull.pdf
    # Note:  marginal errors.
    merr           = np.sqrt(np.diag(ihess))

    ##
    rflux          = {}
    
    for cam in cams:
        wave        = postage[cam].wave
        res         = Resolution(postage[cam].R)
    
        rflux[cam]  = doublet_obs(z, wave, res, continuum=0.0, sigmav=v, r=r, lineida=lineida, lineidb=lineidb)
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

    petal    =   '5'
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
            
    # [Group][Line Index][Camera].
    postages = cframe_postage(cframes, fiber, rrz)

    ncol = 23
    
    ##  Plot.                                                                                                                                                                                                                                 
    fig, axes = plt.subplots(len(ugroups), ncol, figsize=(50,10))

    for u in ugroups:
        index = 0

        for i, x in enumerate(postages[u]):
            for cam in postages[u][x]:
                axes[u,index].axvline((1. + rrz) * lines['WAVELENGTH'][x], c='k', lw=0.5)
                axes[u,index].plot(postages[u][x][cam].wave, postages[u][x][cam].flux, alpha=0.5, lw=0.75, c=colors[cam[0]])
                axes[u,index].set_title(lines['NAME'][x])

                index += 1
                
    fig.suptitle('Fiber {} of petal {} with redshift {:2f}'.format(fiber, petal, rrz), y=1.02)

    plt.tight_layout()

    pl.savefig('postages.pdf')

    pl.clf()
    
    # Starting guess: [z, v, r, lnA]
    x0            = np.array([rrz + .5e-3, 52., 0.7, 6.0])
    result, rflux = doublet_fit(x0, rrz, rrzerr, postages[3], lineida=6, lineidb=7, plot=True)

    print(result[0])
    
    print('\n\nDone.\n\n')
