import time 
import numpy as np
import pickle
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
from   matchedtemp_lineflux import matchedtemp_lineflux
from   doublet_obs import doublet_obs
from   plot_postages import plot_postages


width  = 50.
cwidth = 25.

def cframe_postage(petal_cframes, fiber, redshift, ipostage=True, printit=False):    
    '''
    Given a redshift, cframe (extracted wave, res, flux, ivar) return 
    chi sq. for a doublet line model of given parameters, e.g. line flux.
    '''

    postages  = {}
    ipostages = {}

    for u in ugroups:
        postages[u] = {}
        ipostages[u] = {}
        
    sample     = lines[lines['MASKED'] == 0]
        
    for i, line in enumerate(sample['WAVELENGTH']):        
        center = (1. + redshift) * line
        limits = center + np.array([-width, width])

        name   = sample['NAME'][i]
        group  = sample['GROUP'][i]
        lratio = sample['LINERATIO'][i]
        
        postages[group][sample['INDEX'][i]] = {}
        ipostages[group][sample['INDEX'][i]] = {}
        
        for cam in petal_cframes.keys():        
            wave   = petal_cframes[cam].wave
            inwave = (wave > limits[0]) & (wave < limits[1])
            
            isin   = (wave.min() < center) & (center < wave.max())
            
            if isin:
                if printit:
                    print('Reduced LINEID {:2d}:  {:16s} for {} at redshift {:.2f} ({:.3f} to {:.3f}).'.format(sample['INDEX'][i], sample['NAME'][i], cam, redshift, limits[0], limits[1]))

                res       = petal_cframes[cam].resolution_data[fiber,:,:]
                flux      = petal_cframes[cam].flux[fiber,:]
                ivar      = petal_cframes[cam].ivar[fiber,:]
                mask      = petal_cframes[cam].mask[fiber,:]

                unmask    = mask == 0
                nelem     = np.count_nonzero(unmask[inwave])

                if nelem == 0:
                    continue
                
                continuum = (wave > limits[0]) & (wave < limits[1]) & ((wave < (limits[0] + cwidth)) | (wave > (limits[1] - cwidth)))
                continuum = np.median(flux[continuum])

                instance  = Spectrum(wave[inwave], flux[inwave] - continuum, ivar[inwave], mask=mask[inwave], R=res[:,inwave])

                # matchedtemp_lineflux(z, wave, res, flux, ivar, mask, continuum=0.0, sigmav=50.0, r=0.7, linea=3726.032, lineb=3728.815)
                instance.meta['LINE']     = name.ljust(15)
                instance.meta['LINEID']   = sample['INDEX'][i]
                instance.meta['LINEFLUX'] = matchedtemp_lineflux(redshift, instance.wave, Resolution(instance.R), instance.flux, instance.ivar, instance.mask, sigmav=90.0, r=0.0, linea=0.0, lineb=line)

                # print('{} {} {:.2e} {:.2e}'.format(name.ljust(15), cam, instance.meta['LINEFLUX'][0], instance.meta['LINEFLUX'][1]))
                
                postages[group][sample['INDEX'][i]][cam]       = instance

                if ipostage:
                    ipostages[group][sample['INDEX'][i]][cam]  = doublet_obs(redshift, instance.wave, instance.wave, Resolution(instance.R), continuum=0.0, sigmav=90.0, r=0.0, linea=0.0, lineb=line)
                    ipostages[group][sample['INDEX'][i]][cam] *= instance.meta['LINEFLUX'][0]
                    ipostages[group][sample['INDEX'][i]][cam]  = Spectrum(instance.wave, ipostages[group][sample['INDEX'][i]][cam], instance.ivar, mask=instance.mask, R=instance.R)
                
    for u in ugroups:
        keys = list(postages[u].keys())
        
        for i in keys:
            if not bool(postages[u][i]):
                del postages[u][i]

        if not bool(postages[u]):
            del postages[u]

    if ipostage:
        print('\n\n')
        
        for u in postages.keys():
            print()
            for lineid in postages[u].keys():
                for cam in postages[u][lineid].keys():
                    name  = postages[u][lineid][cam].meta['LINE']
                    index = postages[u][lineid][cam].meta['LINEID']
                    lflux = postages[u][lineid][cam].meta['LINEFLUX'] 
                
                    print('{} ({: 2d}) {} {:.2f} +- {:.2f} ({:.2f})'.format(name, index, cam, lflux[0], lflux[1], np.log(np.maximum(lflux[0], 1.e-10))))
            
    return  postages, ipostages


if __name__ == '__main__':
    import os
    import pylab as pl
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

    rrz       = meta['REDSHIFT'][115]

    print(rrz)
    
    print('\n\nOIIFLUX: {:.2f}'.format(1.e17 * objmeta['OIIFLUX'][115]))
    print('HBETAFLUX: {:.2f}\n\n'.format(objmeta['HBETAFLUX'][115]))
    
    start     = time.time()
    
    postages, ipostages = cframe_postage(cframes, fiber, rrz)

    end       = time.time()

    print('\n\nFinised postages in {:.2f} seconds.'.format(end - start))

    fig       = plot_postages(postages, ipostages, petal, fiber, rrz, tid)
    fig.savefig('ipostages.pdf', bbox_inches='tight')
    
    ##  Plot.
    fig, axes = plt.subplots(len(ugroups), 10, figsize=(25,10))

    for u in np.unique(list(postages.keys())):
        index = 0

        for _, x in enumerate(list(postages[u].keys())):
            for cam in postages[u][x]:
                axes[u,index].axvline((1. + rrz) * lines['WAVELENGTH'][x], c='k', lw=0.5)
                axes[u,index].plot(postages[u][x][cam].wave, postages[u][x][cam].flux, alpha=0.5, lw=0.75, c=colors[cam[0]])
                axes[u,index].set_title(lines['NAME'][x])

                index += 1
        
    fig.suptitle('Fiber {} of petal {} with redshift {:2f}'.format(fiber, petal, rrz), y=1.02)

    plt.tight_layout()
    
    pl.show()
    
    print('\n\nDone.\n\n')
