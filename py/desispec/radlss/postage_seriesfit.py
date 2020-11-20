import time
import numpy as np
import scipy
import iminuit

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
from   doublet_postagefit import doublet_obs, doublet_chi2, doublet_fit
from   desispec.frame import Spectrum
from   autograd import grad
from   twave import twave
from   astropy.table import Table, vstack  
from   doublet_priors import gaussian_prior, jeffreys_prior
from   plot_postages import plot_postages


def series_fit(rrz, rrzerr, postages, group=3, mpostages=None, printit=False):
    '''
    '''

    postages         = postages[group]
    
    lineids          = list(postages.keys())
    
    singlets         = lines[lineids][lines['DOUBLET'][lineids] == -99]
    doublets         = lines[lineids][lines['DOUBLET'][lineids] >=   0]

    # If only one line is present in the doublet, treat as a singlet.
    udoublets, cnts  = np.unique(doublets['DOUBLET'], return_counts=True)

    for udub, cnt in zip(udoublets, cnts):
        if cnt == 1:
            singlets = vstack([singlets, doublets[doublets['DOUBLET'] == udub]])
            doublets = doublets[doublets['DOUBLET'] != udub]
             
    nsinglet         = len(singlets)
    ndoublet         = np.int(len(doublets) / 2)

    if printit:
        if len(singlets) > 0:
            print(singlets)

        if len(doublets) > 0:
            print(doublets)
    
    def _X2(x):  
        z             = x[0]
        v             = x[1]

        result        = 0.0

        for i, singlet in enumerate(singlets):
            lineidb   = singlet['INDEX']
            lineb     = singlet['WAVELENGTH']

            if singlet['MASKED'] == 1:
                continue
            
            postage   = postages[lineidb]
            cams      = list(postage.keys())

            lnA       = x[i + 2]
            line_flux = np.exp(lnA)

            # print(lineidb, cams)
            
            for cam in cams:
                wave  = postage[cam].wave
                res   = Resolution(postage[cam].R)
                flux  = postage[cam].flux
                ivar  = postage[cam].ivar
                mask  = postage[cam].mask

                tw    = twave(wave.min(), wave.max())
                
                # lineida irrelevant when line ratio, r, is zero.
                _, rflux, X2, _ = doublet_chi2(z, tw, wave, res, flux, ivar, mask, continuum=0.0, sigmav=v, r=0.0, line_flux=line_flux, linea=0.0, lineb=lineb)
                
                result += X2
        
        for i in np.arange(ndoublet):
            doublet   = doublets[doublets['DOUBLET'] == i] 

            if (doublet['MASKED'][0] == 1) | (doublet['MASKED'][1] == 1):
                continue

            lineida   = doublet['INDEX'][0]
            lineidb   = doublet['INDEX'][1] 
            
            linea     = doublet['WAVELENGTH'][0]
            lineb     = doublet['WAVELENGTH'][1]

            postage   = postages[lineidb]
            cams      = list(postage.keys())

            lnA       = x[2 * i + 2 + nsinglet]
            line_flux = np.exp(lnA)
            
            r         = x[2 * i + 3 + nsinglet]

            for cam in cams:
                wave  = postage[cam].wave
                res   = Resolution(postage[cam].R)
                flux  = postage[cam].flux
                ivar  = postage[cam].ivar
                mask  = postage[cam].mask

                tw    = twave(wave.min(), wave.max())

                _, _, X2, _ = doublet_chi2(z, tw, wave, res, flux, ivar, mask, continuum=0.0, sigmav=v, r=r, line_flux=line_flux, linea=linea, lineb=lineb)

                # print(lineidb, cam, z, v, r, line_flux, X2)
                
                result += X2
            
        return  result

    def _res(x):
        z             = x[0]
        v             = x[1]

        result        = 0.0

        residuals     = []
        
        for i, singlet in enumerate(singlets):
            lineidb   = singlet['INDEX']
            lineb     = singlet['WAVELENGTH']
            
            if singlet['MASKED'] == 1:
                continue

            postage   = postages[lineidb]
            cams      = list(postage.keys())

            lnA       = x[i + 2]
            line_flux = np.exp(lnA)

            for cam in cams:
                wave  = postage[cam].wave
                res   = Resolution(postage[cam].R)
                flux  = postage[cam].flux
                ivar  = postage[cam].ivar
                mask  = postage[cam].mask == 0

                tw    = twave(wave.min(), wave.max())

                # lineida irrelevant when line ratio, r, is zero.
                rflux      = doublet_obs(z, tw, wave, res, continuum=0.0, sigmav=v, r=0.0, linea=0.0, lineb=lineb)
                rflux     *= line_flux

                res        = np.sqrt(ivar[mask]) * (flux[mask] - rflux[mask])
                
                residuals += res.tolist()

        for i in np.arange(ndoublet):
            doublet   = doublets[doublets['DOUBLET'] == i]

            if doublet['MASKED'][0] == 1:
                continue

            linea     = doublet['WAVELENGTH'][0]
            lineb     = doublet['WAVELENGTH'][1]

            lineidb   = doublet['INDEX'][1]
            
            postage   = postages[lineidb]
            cams      = list(postage.keys())

            lnA       = x[2 * i + 2 + nsinglet]
            line_flux = np.exp(lnA)

            r         = x[2 * i + 3 + nsinglet]
            
            for cam in cams:
                wave  = postage[cam].wave
                res   = Resolution(postage[cam].R)
                flux  = postage[cam].flux
                ivar  = postage[cam].ivar
                mask  = postage[cam].mask == 0

                tw    = twave(wave.min(), wave.max())

		# print(lineidb, cam, z, v, r, line_flux, X2)

                rflux      = doublet_obs(z, tw, wave, res, continuum=0.0, sigmav=v, r=r, linea=linea, lineb=lineb)
                rflux     *= line_flux

                res        = np.sqrt(ivar[mask]) * (flux[mask] - rflux[mask])
		
                residuals += res.tolist()

        # Include a parameter prior. 
        nelement   = len(residuals)
        residuals += np.sqrt(2. * mlogprior(x, rrz, rrzerr) / (1. + nelement))

        residuals  = np.array(residuals)

        return  residuals

    def mloglike(x):
        return  _X2(x) / 2.

    def mlogpos(x):
        return  mloglike(x) + mlogprior(x, rrz, rrzerr)

    def gradient(x):
        eps    = np.array([rrzerr / 1.e2, 1.] + [1.0] * nsinglet + [0.2, 0.05] * ndoublet)
        # grad = optimize.approx_fprime(x, mlogpos, epsilon=eps)

        grad   = np.zeros_like(x)

        for i, element in enumerate(x):
            step     = np.zeros_like(x)
            step[i]  = eps[i]
            grad[i]  = (mlogpos(x + step) - mlogpos(x - step)) / (2. * eps[i])
        
        return  grad
        
    # x0: [z, v, [ln(line flux) fxor all singlets], [[ln(line flux), line ratio] for all doublets]]                                                                                                                                         
    # print('\n\nBeginning optimisation for line group {}.'.format(group))

    for sig0, r0, lnA0 in zip([50.0, 50.0, 100.], [0.7, 0.7, 0.9], [5.0, 4.0, 5.0]):
        x0 = [rrz, sig0] + [lnA0] * nsinglet + [lnA0, r0] * ndoublet
        x0 = np.array(x0)

        # print('{} \t {} \t {:.6e} \t {:.6e}'.format(x0, gradient(x0), mlogpos(x0), _X2(x0)))

        break
    
    # Parameters which minimize f, i.e., f(xopt) == fopt.
    # Minimum value.
    # Value of gradient at minimum, fprime(xopt), which should be near 0.
    # Value of 1/fdoubleprime(xopt), i.e., the inverse Hessian matrix.
    # Number of function_calls made.
    # Number of gradient calls made.
    # Maximum number of iterations exceeded. 2 : Gradient and/or function calls not changing. 3 : NaN result encountered. 
    # The value of xopt at each iteration. Only returned if retall is True.
    
    # See:  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_bfgs.html
    # epsilon=1.4901161193847656e-08; norm=-np.inf

    # method='CG'
    # result     = scipy.optimize.minimize(mlogpos, x0, method='Nelder-Mead', options = {'disp': True, 'maxiter': 5000}, tol=1e-5)
    # bestfit    = result.x

    # fprime=gradient;  gtol=1e-08, maxiter=500
    # result     = scipy.optimize.fmin_bfgs(mlogpos, x0, gtol=1e-3, maxiter=500, full_output=True)
    # bestfit    = result[0]
    # ihess      = result[3]
    # print(bestfit)
    
    result       = scipy.optimize.least_squares(_res, x0, verbose=0, ftol=1.e-8, max_nfev=50)
    bestfit      = result.x

    # https://astrostatistics.psu.edu/su11scma5/HeavensLecturesSCMAVfull.pdf
    # Note:  marginal errors.
    # merr         = np.sqrt(np.diag(ihess))
    
    ##
    if mpostages is None:
        mpostages  = {}

    mpostages[group] = {}

    bestfit_z     = bestfit[0]
    bestfit_v     = bestfit[1]

    # print(doublets)

    series_params = ['z', 'v']
    
    for i, singlet in enumerate(singlets):
        lineidb   = singlet['INDEX']
        lineb     = singlet['WAVELENGTH']
        
        if singlet['MASKED'] == 1:
            continue

        postage   = postages[lineidb]
        cams      = list(postage.keys())

        lnA       = bestfit[i + 2]
        line_flux = np.exp(lnA)

        mpostages[group][lineidb] = {}
        
        for cam in cams:
            wave  = postage[cam].wave
            res   = Resolution(postage[cam].R)
            ivar  = postage[cam].ivar
            mask  = postage[cam].mask

            tw    = twave(wave.min(), wave.max())
            
            # lineida irrelevant when line ratio, r, is zero. 
            mpostages[group][lineidb][cam]  = doublet_obs(bestfit_z, tw, wave, res, continuum=0.0, sigmav=bestfit_v, r=0.0, linea=0.0, lineb=lineb)
            mpostages[group][lineidb][cam] *= np.exp(lnA)
            mpostages[group][lineidb][cam]  = Spectrum(wave, mpostages[group][lineidb][cam], ivar, mask=mask, R=res)

        series_params += ['lnA_{:d}'.format(lineidb)]
            
    for i in np.arange(ndoublet):
        doublet   = doublets[doublets['DOUBLET'] == i]

        lineida   = doublet['INDEX'][0]
        lineidb   = doublet['INDEX'][1]

        linea     = doublet['WAVELENGTH'][0]
        lineb     = doublet['WAVELENGTH'][1]

        if (doublet['MASKED'][0] == 1) | (doublet['MASKED'][1] == 1):
            continue
        
        postage   = postages[lineidb]
        cams      = list(postage.keys())

        lnA       = bestfit[2 * i + 2 + nsinglet]
        line_flux = np.exp(lnA)

        r         = bestfit[2 * i + 3 + nsinglet]
        
        mpostages[group][lineida] = {}
        mpostages[group][lineidb] = {}
        
        for cam in cams:
            wave  = postage[cam].wave
            res   = Resolution(postage[cam].R)
            ivar  = postage[cam].ivar
            mask  = postage[cam].mask

            tw    = twave(wave.min(), wave.max())

            mpostages[group][lineidb][cam]  = doublet_obs(bestfit_z, tw, wave, res, continuum=0.0, sigmav=bestfit_v, r=r, linea=linea, lineb=lineb)
            mpostages[group][lineidb][cam] *= np.exp(lnA)

            mpostages[group][lineida][cam]  = Spectrum(wave, mpostages[group][lineidb][cam], ivar, mask=mask, R=res)
            mpostages[group][lineidb][cam]  = mpostages[group][lineida][cam]

        series_params += ['lnA_{:d}-{:d}'.format(lineida, lineidb), 'r_{:d}-{:d}'.format(lineida, lineidb)]
            
    return  series_params, result, mpostages

def plot_postages(postages, mpostages, petal, fiber, rrz, tid):
    import matplotlib.pyplot as plt

    
    ncol      = 8
    fig, axes = plt.subplots(len(ugroups), ncol, figsize=(20,10))
    
    colors    = {'b': 'b', 'r': 'g', 'z': 'r'}

    stop      = False
    
    for u in list(postages.keys()):
        index = 0

        for i, x in enumerate(list(postages[u].keys())):
            for cam in postages[u][x]:
                if lines['MASKED'][x] == 1:
                    continue

                if not stop:                    
                    axes[u,index].axvline((1. + rrz) * lines['WAVELENGTH'][x], c='k', lw=0.5)
                    axes[u,index].plot(postages[u][x][cam].wave, postages[u][x][cam].flux, alpha=0.5, lw=0.75, c=colors[cam[0]])
                    axes[u,index].plot(mpostages[u][x][cam].wave, mpostages[u][x][cam].flux, alpha=0.5, lw=0.75, c='k', linestyle='--')
                    axes[u,index].set_title(lines['NAME'][x])

                    index += 1

                if index == (ncol - 1):
                    stop = True
                
    fig.suptitle('Fiber {} of petal {}: targetid {} with redshift {:2f}'.format(fiber, petal, tid, rrz), y=0.925)

    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    return  fig 


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
    tid      = zbest['TARGETID'][fiber]
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
        
    rrz               = meta['REDSHIFT'][115]

    # [Group][Line Index][Camera].
    postages          = cframe_postage(cframes, fiber, rrz)

    mpostages         = {}

    groups            = list(postages.keys())

    for group in groups:
        series_params, result, mpostages = series_fit(rrz, rrzerr, postages, group=group, mpostages=mpostages)

    start             = time.time() 
    
    # [Group][Line Index][Camera].
    postages          = cframe_postage(cframes, fiber, rrz)
    
    mpostages         = {}

    groups            = list(postages.keys())
    
    for group in groups:
        series_params, result, mpostages = series_fit(rrz, rrzerr, postages, group=group, mpostages=mpostages)

        print(len(result.x), series_params)
        
    end               = time.time()

    print('\n\nMinimised in {:.2f} seconds.'.format(end - start))
    
    ## 
    fig       = plot_postages(postages, mpostages, petal, fiber, rrz, tid)
    fig.savefig('postages.pdf', bbox_inches='tight')

    print('\n\nDone.\n\n')
