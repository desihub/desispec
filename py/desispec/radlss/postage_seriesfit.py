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
from   doublet_postagefit import doublet_obs, doublet_chi2, doublet_fit
from   desispec.frame import Spectrum


def emission_fit(rrz, rrzerr, postages, group=3, mpostages=None):
    '''
    '''

    postages         = postages[group]
    
    lineids          = list(postages.keys())
    
    singlets         = lines[lineids][lines['DOUBLET'][lineids] == -99]
    doublets         = lines[lineids][lines['DOUBLET'][lineids] >=   0]

    nsinglet         = len(singlets)
    ndoublet         = np.int(len(doublets) / 2)

    print(singlets)
    print(doublets)
    
    def _X2(x):  
        z             = x[0]
        v             = x[1]

        result        = 0.0

        # import pylab as pl
        
        for i, singlet in enumerate(singlets):
            lineidb   = singlet['INDEX']
            
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

                # lineida irrelevant when line ratio, r, is zero.
                _, rflux, X2, _ = doublet_chi2(z, wave, res, flux, ivar, mask, continuum=0.0, sigmav=v, r=0.0, line_flux=line_flux, lineida=0, lineidb=lineidb)

                '''
                import pylab as pl
                
                pl.plot(wave, flux, label='Data')
                pl.plot(wave, rflux, label='Model')
                pl.legend()
                pl.show()
                '''
                
                # print(lineidb, cam, z, v, line_flux, X2)
                # print(flux)
                # print(mask)
                # print(rflux)
                
                result += X2

        for i in np.arange(ndoublet):
            doublet   = doublets[doublets['DOUBLET'] == i] 
            
            lineida   = doublet['INDEX'][0]
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
                mask  = postage[cam].mask

                _, _, X2, _ = doublet_chi2(z, wave, res, flux, ivar, mask, continuum=0.0, sigmav=v, r=r, line_flux=line_flux, lineida=lineida, lineidb=lineidb)

                result += X2
            
        return  result

    def mloglike(x):
        return  _X2(x) / 2.

    def mlogpos(x):
        return  mloglike(x) # + mlogprior(x, rrz, rrzerr)

    def gradient(x):
        # eps = 1.e-8
        eps   = np.array([1.e-4, 10.] + [0.1] * nsinglet + [0.1, 0.025] * ndoublet)
        
        return  optimize.approx_fprime(x, mlogpos, eps)

    # x0: [z, v, [ln(line flux) fxor all singlets], [[ln(line flux), line ratio] for all doublets]]                                                                                                                                         
    print('\n\nBeginning optimisation for line group {}.'.format(group))

    for sig0, r0, lnA0 in zip([150.0, 50.0, 100.], [0.5, 0.7, 0.9], [3.5, 4.0, 5.0]):
        x0 = [rrz + 5.e-4, sig0] + [lnA0] * nsinglet + [lnA0, r0] * ndoublet
        x0 = np.array(x0)

        # print('{} \t {:.6e} \t {:.6e}'.format(gradient(x0), mlogpos(x0), _X2(x0)))

        break

    print(x0)
    
    # Parameters which minimize f, i.e., f(xopt) == fopt.
    # Minimum value.
    # Value of gradient at minimum, fprime(xopt), which should be near 0.
    # Value of 1/fdoubleprime(xopt), i.e., the inverse Hessian matrix.
    # Number of function_calls made.
    # Number of gradient calls made.
    # Maximum number of iterations exceeded. 2 : Gradient and/or function calls not changing. 3 : NaN result encountered. 
    # The value of xopt at each iteration. Only returned if retall is True.
    
    # See:  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_bfgs.html
    result         = scipy.optimize.fmin_bfgs(mlogpos, x0, fprime=gradient, gtol=1e-05, norm=-np.inf, epsilon=1.4901161193847656e-08, maxiter=None, full_output=True, disp=1, retall=1, callback=None)    

    bestfit        = result[0]
    ihess          = result[3]

    # https://astrostatistics.psu.edu/su11scma5/HeavensLecturesSCMAVfull.pdf
    # Note:  marginal errors.
    merr           = np.sqrt(np.diag(ihess))

    ##
    if mpostages is None:
        mpostages  = {}

    mpostages[group] = {}


    bestfit_z     = bestfit[0]
    bestfit_v     = bestfit[1]

    # print(doublets)
    
    for i, singlet in enumerate(singlets):
        lineidb   = singlet['INDEX']
        
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
            
            # lineida irrelevant when line ratio, r, is zero. 
            mpostages[group][lineidb][cam]  = doublet_obs(bestfit_z, wave, res, continuum=0.0, sigmav=bestfit_v, r=0.0, lineida=0, lineidb=lineidb)
            mpostages[group][lineidb][cam] *= np.exp(lnA)
            mpostages[group][lineidb][cam]  = Spectrum(wave, mpostages[group][lineidb][cam], ivar, mask=mask, R=res)
    
    for i in np.arange(ndoublet):
        doublet   = doublets[doublets['DOUBLET'] == i]

        lineida   = doublet['INDEX'][0]
        lineidb   = doublet['INDEX'][1]
        
        postage   = postages[lineidb]
        cams      = list(postage.keys())

        lnA       = bestfit[2 * i + 2 + nsinglet]
        line_flux = np.exp(lnA)

        r         = bestfit[2 * i + 3 + nsinglet]

        mpostages[group][lineidb] = {}
        
        for cam in cams:
            wave  = postage[cam].wave
            res   = Resolution(postage[cam].R)
            ivar  = postage[cam].ivar
            mask  = postage[cam].mask

            mpostages[group][lineidb][cam]  = doublet_obs(bestfit_z, wave, res, continuum=0.0, sigmav=bestfit_v, r=r, lineida=lineida, lineidb=lineidb)
            mpostages[group][lineidb][cam] *= np.exp(lnA)

            mpostages[group][lineidb][cam]  = Spectrum(wave, mpostages[group][lineidb][cam], ivar, mask=mask, R=res)
    
    return  result, mpostages
    

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
            
    # [Group][Line Index][Camera].
    postages          = cframe_postage(cframes, fiber, rrz)


    mpostages         = {}

    for group in list(postages.keys()):
        result, mpostages = emission_fit(rrz, rrzerr, postages, group=group, mpostages=mpostages)

        break

    
    ## 
    ncol   = 23

    for toplot in [postages, mpostages]:    
        fig, axes = plt.subplots(len(ugroups), ncol, figsize=(50,10))

        for u in list(toplot.keys()):
            index = 0

            for i, x in enumerate(toplot[u]):
                for cam in toplot[u][x]:
                    axes[u,index].axvline((1. + rrz) * lines['WAVELENGTH'][x], c='k', lw=0.5)
                    axes[u,index].plot(toplot[u][x][cam].wave, toplot[u][x][cam].flux, alpha=0.5, lw=0.75, c=colors[cam[0]])
                    axes[u,index].set_title(lines['NAME'][x])

                    index += 1
                
    fig.suptitle('Fiber {} of petal {} with redshift {:2f}'.format(fiber, petal, rrz), y=1.02)

    plt.tight_layout()

    pl.savefig('postages.pdf')
    
    print('\n\nDone.\n\n')
