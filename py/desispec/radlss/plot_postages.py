import matplotlib.pyplot as plt

from lines import ugroups, lines

def plot_postages(postages, mpostages, petal, fiber, rrz, tid):
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
