import numpy as np
import matplotlib.pyplot as plt

from   lines import ugroups, lines
from   astropy.convolution import convolve, Box1DKernel

def plot_postages(postages, mpostages, petal, fiber, rrz, tid, results=None):
    ncol      = 8
    fig, axes = plt.subplots(len(ugroups), ncol, figsize=(20,10))
    
    colors    = {'b': 'b', 'r': 'g', 'z': 'r'}

    stop      = False

    # Group.
    for u in list(postages.keys()):
        index = 0

        print('\n\n----  Plotting {} series  ----'.format(u))
        
        # Lineid.
        for i, x in enumerate(list(postages[u].keys())):
            # Camera.
            for cam in postages[u][x]:
                if lines['MASKED'][x] == 1:
                    continue

                sstr = '    {: 4d}    {}'.format(x, cam.ljust(4))
                
                if not stop:                    
                    if results is not None:
                        if u in results.keys():
                            bestz = results[u].x[0]
                            axes[u,index].axvline((1. + bestz) * lines['WAVELENGTH'][x], c='k', lw=0.2)

                    axes[u,index].axvline((1. + rrz) * lines['WAVELENGTH'][x], c='r', lw=0.2, linestyle='--')
                        
                    axes[u,index].plot(postages[u][x][cam].wave, postages[u][x][cam].flux, alpha=1.00, lw=0.5, c=colors[cam[0]])
                    # axes[u,index].plot(postages[u][x][cam].wave, convolve(postages[u][x][cam].flux, Box1DKernel(5)), alpha=1.00, lw=0.5, c=colors[cam[0]])                    

                    axes[u,index].set_title('{} ({})'.format(lines['NAME'][x], lines['INDEX'][x]))

                    mpostage = None
                    
                    try:                        
                        mpostage = mpostages[u][x][cam]
                        
                    except:
                        sstr += '\t ** Failed to retrieve mpostage. **'

                    if mpostage is not None:
                        axes[u,index].plot(mpostage.wave, mpostage.flux, alpha=0.5, lw=0.75, c='k', linestyle='--')

                        if not np.allclose(mpostage.wave, postages[u][x][cam].wave):
                            print('Ill-matching wavelength grid for {} {} {}'.format(u, i, cam))

                            print(mpostage.wave)
                            print(postages[u][x][cam].wave)
                            
                    index += 1

                    if index == ncol:
                        stop = True
                        
                print(sstr)

    fig.suptitle('Fiber {} of petal {}: targetid {} with redshift {:2f}'.format(fiber, petal, tid, rrz), y=0.925)

    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    return  fig
