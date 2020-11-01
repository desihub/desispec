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
from   desispec.frame import Spectrum
from   lines import lines, ugroups


def cframe_postage(petal_cframes, fiber, redshift):    
    '''
    Given a redshift, cframe (extracted wave, res, flux, ivar) return 
    chi sq. for a doublet line model of given parameters, e.g. line flux.

    If line flux is None, estimate it first. 
    '''

    postages = {}

    for u in ugroups:
        postages[u] = {}

    for i, line in enumerate(lines['WAVELENGTH']):
        limits = (1. + redshift) * line + np.array([-50., 50.])
        group  = lines['GROUP'][i]

        postages[group][lines['INDEX'][i]]  = {}
        
        for cam in petal_cframes.keys():        
            res    = petal_cframes[cam].resolution_data[fiber,:,:]
            flux   = petal_cframes[cam].flux[fiber,:]
            ivar   = petal_cframes[cam].ivar[fiber,:]
            mask   = petal_cframes[cam].mask[fiber,:]
            wave   = petal_cframes[cam].wave

            inwave = (wave > limits[0]) & (wave < limits[1])

            if np.count_nonzero(inwave):
                print('Reduced LINEID {:2d}:  {:16s} for {} at redshift {:.2f} ({:.3f} to {:.3f}).'.format(lines['INDEX'][i], lines['NAME'][i], cam, redshift, limits[0], limits[1]))
    
                continuum = (wave > limits[0]) & (wave < limits[1]) & ((wave < (limits[0] + 30.)) | (wave > (limits[1] - 30.)))
                continuum = np.median(flux[continuum])
                            
                postages[group][lines['INDEX'][i]][cam] = Spectrum(wave[inwave], flux[inwave] - continuum, ivar[inwave], mask=mask[inwave], R=res[:,inwave])
                
    return  postages


if __name__ == '__main__':
    import os
    import pylab as pl
    import matplotlib.pyplot as plt

    from   astropy.table import Table

    
    petal    = '0'
    fiber    =  7
    
    zbest    =  Table.read(os.path.join('/global/homes/m/mjwilson/andes/', 'tiles', str(67142), str(20200315), 'zbest-{}-{}-{}.fits'.format(petal, 67142, 20200315)), 'ZBEST')
    redshift =  zbest['Z'][fiber]

    cframes  = {}
    colors   = {'b': 'b', 'r': 'g', 'z': 'r'}
    
    for band in ['b', 'r', 'z']:
      cam    = band+petal
         
      # E.g. ~/andes/exposures/20200315/00055642/cframe-b0-00055642.fits
      cframes[cam] = read_frame(findfile('cframe', night=20200315, expid=55642, camera=cam, specprod_dir='/global/homes/m/mjwilson/andes/'))
      
    zbest     = Table.read(os.path.join('/global/homes/m/mjwilson/andes/', 'tiles', str(67142), str(20200315), 'zbest-{}-{}-{}.fits'.format(petal, 67142, 20200315)), 'ZBEST')
      
    postages  = cframe_postage(cframes, fiber, redshift)

    ##  Plot.
    fig, axes = plt.subplots(len(ugroups), 14, figsize=(25,10))

    for u in ugroups:
        index = 0
        
        for i, x in enumerate(postages[u]):
            for cam in postages[u][x]:
                axes[u,index].axvline((1. + redshift) * lines['WAVELENGTH'][x], c='k', lw=0.5)
                axes[u,index].plot(postages[u][x][cam].wave, postages[u][x][cam].flux, alpha=0.5, lw=0.75, c=colors[cam[0]])
                axes[u,index].set_title(lines['NAME'][x])

                index += 1
                
    fig.suptitle('Fiber {} of petal {} with redshift {:2f}'.format(fiber, petal, redshift), y=1.02)

    plt.tight_layout()
    
    pl.show()
    
    print('\n\nDone.\n\n')
