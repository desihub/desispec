import time 
import numpy as np
import scipy

from   scipy             import optimize
from   scipy.optimize    import approx_fprime, minimize, Bounds
from   scipy.stats       import multivariate_normal

from   doublet import doublet
from   desispec.io import read_frame
from   desispec.io.meta import findfile
from   desispec.resolution import Resolution
from   doublet_priors import mlogprior
from   desispec.frame import Spectrum
from   lines import lines, ugroups


width  = 50.
cwidth = 25.

def cframe_postage(petal_cframes, fiber, redshift, printit=False):    
    '''
    Given a redshift, cframe (extracted wave, res, flux, ivar) return 
    chi sq. for a doublet line model of given parameters, e.g. line flux.
    '''

    postages = {}

    for u in ugroups:
        postages[u] = {}

    sample     = lines[lines['MASKED'] == 0]
        
    for i, line in enumerate(sample['WAVELENGTH']):
        limits = (1. + redshift) * line + np.array([-width, width])
        group  = sample['GROUP'][i]

        postages[group][sample['INDEX'][i]] = {}
        
        for cam in petal_cframes.keys():        
            wave   = petal_cframes[cam].wave
            inwave = (wave > limits[0]) & (wave < limits[1])

            if np.count_nonzero(inwave):
                if printit:
                    print('Reduced LINEID {:2d}:  {:16s} for {} at redshift {:.2f} ({:.3f} to {:.3f}).'.format(sample['INDEX'][i], sample['NAME'][i], cam, redshift, limits[0], limits[1]))

                res       = petal_cframes[cam].resolution_data[fiber,:,:]
                flux      = petal_cframes[cam].flux[fiber,:]
                ivar      = petal_cframes[cam].ivar[fiber,:]
                mask      = petal_cframes[cam].mask[fiber,:]
                    
                continuum = (wave > limits[0]) & (wave < limits[1]) & ((wave < (limits[0] + cwidth)) | (wave > (limits[1] - cwidth)))
                continuum = np.median(flux[continuum])
                            
                postages[group][sample['INDEX'][i]][cam] = Spectrum(wave[inwave], flux[inwave] - continuum, ivar[inwave], mask=mask[inwave], R=res[:,inwave])

    for u in ugroups:
        keys = list(postages[u].keys())
        
        for i in keys:
            if not bool(postages[u][i]):
                del postages[u][i]

        if not bool(postages[u]):
            del postages[u]

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

    start    = time.time()
    
    postages  = cframe_postage(cframes, fiber, redshift)

    end       = time.time()

    print(end - start)

    '''
    ##  Plot.
    fig, axes = plt.subplots(len(ugroups), 14, figsize=(25,10))

    for u in np.unique(list(postages.keys())):
        index = 0

        for _, x in enumerate(list(postages[u].keys())):
            for cam in postages[u][x]:
                axes[u,index].axvline((1. + redshift) * lines['WAVELENGTH'][x], c='k', lw=0.5)
                axes[u,index].plot(postages[u][x][cam].wave, postages[u][x][cam].flux, alpha=0.5, lw=0.75, c=colors[cam[0]])
                axes[u,index].set_title(lines['NAME'][x])

                index += 1
        
    fig.suptitle('Fiber {} of petal {} with redshift {:2f}'.format(fiber, petal, redshift), y=1.02)

    plt.tight_layout()
    
    pl.show()
    '''
    print('\n\nDone.\n\n')
