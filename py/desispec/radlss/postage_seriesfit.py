import time
import numpy as np
import scipy
import iminuit
import warnings
import fitsio

from   scipy             import optimize
from   scipy.optimize    import approx_fprime, minimize, Bounds

from   doublet import doublet
from   desispec.io import read_frame
from   desispec.io.meta import findfile
from   desispec.resolution import Resolution
from   doublet_priors import mlogprior
from   cframe_postage import cframe_postage
from   lines import lines, ugroups
from   doublet_postagefit import doublet_obs, doublet_chi2, doublet_fit
from   desispec.frame import Spectrum
from   twave import twave
from   astropy.table import Table, vstack  
from   doublet_priors import gaussian_prior, jeffreys_prior
from   plot_postages import plot_postages

warnings.filterwarnings('error')

def series_fit(rrz, rrzerr, postages, group=3, sig0=41.37, mpostages=None, printit=False, verbose=1):
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
            print(singlets[singlets['MASKED'] == 0])

        if len(doublets) > 0:
            print(doublets[doublets['MASKED'] == 0])
    
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

        prior         = gaussian_prior(z, rrz, rrzerr)
        prior        *= gaussian_prior(v,  90,    50.)
        
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

            prior    *= gaussian_prior(r, 0.7, 0.2)
            
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

        residuals += [np.sqrt(2. * mlogprior(prior) / (1. + nelement))]            
        residuals  = np.array(residuals)
        
        return  residuals

    def mloglike(x):
        raise NotImplementedError()

    def mlogpos(x):
        return  0.5 * np.sum(_res(x)**2.)

    def gradient(x):
        eps    = np.array([rrzerr / 1.e2, 1.] + [1.0] * nsinglet + [0.2, 0.05] * ndoublet)
        grad   = np.zeros_like(x)

        for i, element in enumerate(x):
            step     = np.zeros_like(x)
            step[i]  = eps[i]
            grad[i]  = (mlogpos(x + step) - mlogpos(x - step)) / (2. * eps[i])

        # grad = optimize.approx_fprime(x, mlogpos, epsilon=eps) 
            
        return  grad

    def hessian(x):
        npar   = len(x)

        eps    = 1.49e-08
        epss   = np.array([eps, eps] + [eps] * nsinglet + [eps, eps] * ndoublet)

        # epss = np.array([rrzerr / 1.e2, 5.] + [3.0] * nsinglet + [3.0, 0.1] * ndoublet)
        
        hess   = np.zeros(npar * npar).reshape(npar, npar)
        zero   = np.zeros_like(x)
        
        for j, xj in enumerate(x):
            for i, xi in enumerate(x):
                if i <= j:
                    astep      = np.zeros_like(x)
                    bstep      = np.zeros_like(x)
                    
                    astep[i]   = epss[i]
                    bstep[j]   = epss[j]

                    hess[i,j]  = mlogpos(x + astep + bstep) - mlogpos(x + astep - bstep) - mlogpos(x - astep + bstep) + mlogpos(x - astep - bstep)

                    hess[i,j] /= (4. * epss[i] * epss[j])
                    
                    # Symmetric.
                    hess[j,i]  = hess[i,j]

                    # print('{} {} {:.10e}'.format(i, j, hess[i,j]))
                    
        return  hess
    
    # x0: [z, v, [ln(line flux) fxor all singlets], [[ln(line flux), line ratio] for all doublets]]                                                                                                                                         
    # print('\n\nBeginning optimisation for line group {}.'.format(group))

    x0 = [rrz, sig0]
    
    # Seed initial amplitudes.  
    for i, singlet in enumerate(singlets):
        lineidb   = singlet['INDEX']
        lineb     = singlet['WAVELENGTH']

        if singlet['MASKED'] == 1:
            continue

        postage   = postages[lineidb]
        cams      = list(postage.keys())

        linefluxerr = np.inf
        
        for cam in cams:
            seed            = postage[cam].meta['LINEFLUX']

            if seed[1] < linefluxerr:
                lineflux    = seed[0] 
                linefluxerr = seed[1]
        
        x0       += [np.log(np.maximum(lineflux, 1.e-10))] 
        
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

        linefluxerr = np.inf

        for cam in cams:
            seed            = postage[cam].meta['LINEFLUX']

            if seed[1] < linefluxerr:
                lineflux    = seed[0]
                linefluxerr = seed[1]

        x0 += [np.log(np.maximum(lineflux, 1.e-10)), 0.7]
        
    # x0 = [rrz, sig0] + [lnA0] * nsinglet + [lnA0, r0] * ndoublet
    x0 = np.array(x0)
    
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

    if verbose > 0:
        print('\n\n')
    
    # jac='2-point'
    result       = scipy.optimize.least_squares(_res, x0, verbose=verbose, ftol=1.e-8, max_nfev=50)
    bestfit      = result.x

    hess         = hessian(bestfit)
    # ihess      = np.linalg.inv(hess)
                                          
    # https://astrostatistics.psu.edu/su11scma5/HeavensLecturesSCMAVfull.pdf
    # Note:  marginal errors.
    # merr       = np.sqrt(np.diag(ihess))
    
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

        assert  lineida < lineidb
        
        linea     = doublet['WAVELENGTH'][0]
        lineb     = doublet['WAVELENGTH'][1]

        if (doublet['MASKED'][0] == 1) | (doublet['MASKED'][1] == 1):
            continue
        
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

            tw    = twave(wave.min(), wave.max())

            mpostages[group][lineidb][cam]  = doublet_obs(bestfit_z, tw, wave, res, continuum=0.0, sigmav=bestfit_v, r=r, linea=linea, lineb=lineb)
            mpostages[group][lineidb][cam] *= np.exp(lnA)

            mpostages[group][lineidb][cam]  = Spectrum(wave, mpostages[group][lineidb][cam], ivar, mask=mask, R=res)

        mpostages[group][lineida] = {}
        
        postage   = postages[lineida]
        cams      = list(postage.keys())

        for cam in cams:
            wave  = postage[cam].wave
            res   = Resolution(postage[cam].R)
            ivar  = postage[cam].ivar
            mask  = postage[cam].mask

            tw    = twave(wave.min(), wave.max())

            mpostages[group][lineida][cam]  = doublet_obs(bestfit_z, tw, wave, res, continuum=0.0, sigmav=bestfit_v, r=r, linea=linea, lineb=lineb)
            mpostages[group][lineida][cam] *= np.exp(lnA)

            mpostages[group][lineida][cam]  = Spectrum(wave, mpostages[group][lineida][cam], ivar, mask=mask, R=res)

        series_params += ['lnA_{:d}-{:d}'.format(lineida, lineidb), 'r_{:d}-{:d}'.format(lineida, lineidb)]

    print('\n\nInitialising at:')
    print(series_params)
    print(x0)
    print('\n\nBest fit found at (status {}):'.format(result.status))
    print(result.x)
    print('\n\nGradient:')
    print(result.grad)
    print('\n\nHessian:')
    print(np.diag(hess))
    
    return  series_params, result, mpostages


if __name__ == '__main__':
    import os
    import pickle
    import argparse
    import pylab as pl
    import astropy.io.fits as fits
    import matplotlib.pyplot as plt
    
    from   astropy.table import Table
    from   astropy.convolution import convolve, Box1DKernel
    
    # 12: Failed to retrieve mpostages: 12.
    # 14: Failed to fit OI. 

    parser   = argparse.ArgumentParser()
    parser.add_argument('--fiber', type=int, default=None, help='...')
    parser.add_argument('--petal', type=str, default='9')
    parser.add_argument('--targetid', type=int, default=None) 
    parser.add_argument('--plotspectra', type=int, default=0, help='...')
    parser.add_argument('--night', type=str, default='20200303')
    parser.add_argument('--expid', type=np.int, default=52935)
    parser.add_argument('--tileid', type=str, default='70500')
    
    args     = parser.parse_args()
    
    petal    = args.petal
    fiber    = args.fiber
    tid      = args.targetid
    night    = args.night
    expid    = args.expid
    tileid   = args.tileid

    if (args.fiber is not None) and (args.targetid is not None):
        raise ValueError('Specify either fiber or targetid.')
    
    cframes  = {}
    colors   = {'b': 'b', 'r': 'g', 'z': 'r'}
    '''
    with open(os.environ['CSCRATCH'] + '/radlss/test/ensemble/template-elg-ensemble-meta.fits', 'rb') as handle:
        meta = pickle.load(handle)['ELG']

    with open(os.environ['CSCRATCH'] + '/radlss/test/ensemble/template-elg-ensemble-flux.fits', 'rb') as handle:
        flux = pickle.load(handle)['ELG']

    with open(os.environ['CSCRATCH'] + '/radlss/test/ensemble/template-elg-ensemble-objmeta.fits', 'rb') as handle:
        objmeta = pickle.load(handle)['ELG']
    ''' 
    for band in ['b', 'r', 'z']:
        cam = band + petal

        # E.g. ~/andes/exposures/20200315/00055642/cframe-b0-00055642.fits
        fpath = findfile('cframe', night=night, expid=expid, camera=cam, specprod_dir='/global/homes/m/mjwilson/andes/')

        fibermap     = fitsio.read(fpath, ext='FIBERMAP')

        if fiber is not None:
            tid = fibermap['TARGETID'][fiber]
            status = fibermap['FIBERSTATUS'][fiber]

        else:
            fiber = fibermap['TARGETID'] == args.targetid

            if np.count_nonzero(fiber) > 0:                
                fiber = np.arange(len(fibermap['TARGETID']))[fiber][0]

                print(fiber)
                
            else:
                raise ValueError('TARGETID {} not present in petal {} of exp. {}'.format(args.targetid, args.petal, expid))
            
            status = fibermap['FIBERSTATUS'][fiber]
            
        cframes[cam] = read_frame(fpath)        
        # cframes[cam].flux[fiber,:] = flux[band][115]

        print('Fetched {}.'.format(fpath))

    if status > 0:
        print('\n\nFIBERSTATUS flagged. Done.\n\n')

        exit(0)
        
    zbest    = Table.read(os.path.join('/global/homes/m/mjwilson/andes/', 'tiles', tileid, night, 'zbest-{}-{}-{}.fits'.format(petal, tileid, night)), 'ZBEST')
    row      = zbest['TARGETID'] == tid
    rrz      = zbest['Z'].data[row][0]
    rrzerr   = zbest['ZERR'].data[row][0]
    
    if args.plotspectra:
        fig, ax = plt.subplots(1, 1, figsize=(10,5))
    
        for cam in cframes.keys():        
            ax.plot(cframes[cam].wave, convolve(cframes[cam].flux[fiber,:], Box1DKernel(5), boundary='extend'), lw=0.5, alpha=0.5)
        
        for line in lines:
            ax.axvline((1. + rrz) * line['WAVELENGTH'], c='k', alpha=0.25, lw=0.5)

            if ((1. + rrz) * line['WAVELENGTH'] > 4.e3):
                ax.text((1. + rrz) * line['WAVELENGTH'], np.random.uniform(3., 10.), line['INDEX'], fontsize=6, horizontalalignment='center', verticalalignment='center', rotation=90)
                
        ax.set_title('fiber: {} targetid: {} redshift: {:.4f}'.format(fiber, tid, rrz))
        ax.set_xlim(4.e3, 8.e3)

        # top=10.0
        ax.set_ylim(bottom=-0.5)

        pl.show()
        
        exit(0)
    
    # rrz = meta['REDSHIFT'][115]
    '''
    # [Group][Line Index][Camera].
    postages, ipostages = cframe_postage(cframes, fiber, rrz)

    mpostages         = {}

    groups            = list(postages.keys())
    
    for group in groups:
        series_params, result, mpostages = series_fit(rrz, rrzerr, postages, group=group, mpostages=mpostages, printit=False)
    '''
    start = time.time() 
    
    # [Group][Line Index][Camera].
    postages, ipostages = cframe_postage(cframes, fiber, rrz)
    
    mpostages = {}

    groups = list(postages.keys())

    results = {}

    for group in groups:
        #try:
        series_params, result, mpostages = series_fit(rrz, rrzerr, postages, group=group, mpostages=mpostages, printit=False)
        results[group] = result
        '''    
        except Exception as e:
            results[group] = None
        
            print('\n\nFailed with linefit of group: {}.'.format(group))
            print('\n\n')
            print(e)
            print('\n\n')
        ''' 
    end = time.time()
    
    print('\n\nMinimised in {:.2f} seconds.'.format(end - start))

    fig = plot_postages(postages, ipostages, petal, fiber, rrz, tid, results=results)
    fig.savefig('ipostages.pdf', bbox_inches='tight')

    pl.show()
    
    ## 
    fig = plot_postages(postages, mpostages, petal, fiber, rrz, tid, results=results)
    fig.savefig('mpostages.pdf', bbox_inches='tight')

    pl.show()
    
    print('\n\nDone (fiber: {}, targetid: {}).\n\n'.format(fiber, tid))
