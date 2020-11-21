import numpy as np


def air2vac(wave):
    w   = np.array(wave, copy=True)
    vac = w[w >= 2000.]

    for iter in range(2):
        sigma2 = (1.0e4 / vac)*(1.0e4 / vac)
        fact   = 1.0 + 5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/(57.362 - sigma2)
        vac    = w[w >= 2000.] * fact

    w[w >= 2000.] = vac
        
    return  w


if __name__ == '__main__':
    import pylab as pl
    
    wave = np.arange(0.0, 1.e4, 0.8)

    pl.semilogy(wave, airtovac(wave) / wave)
    pl.show()

    print('\n\nDone.\n\n')
