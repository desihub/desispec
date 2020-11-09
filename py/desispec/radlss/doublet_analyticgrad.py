import numpy as np

from doublet import doublet 
from twave   import twave

# Obs. model = Resolution * [exp(\tilde A) * \tilde M].
def dMdlnA(z, v, r, lnA, linea=3726.032, lineb=3728.815, model=None):
    if model is None:
        model = doublet(z, twave, sigmav=v, r=r, linea=3726.032, lineb=3728.815)

    return model

def dMdr(z, v, r, lnA, linea=3726.032, lineb=3728.815, model=None):
    result  = r / (1. + r)
    result /= np.sqrt(2. * np.pi)
    result /= v**2.
    result *= np.exp(-(twave  - linea * (1. + z) / np.sqrt(2.) / v)**2.)

    return result


if __name__ == '__main__':
    z    = 1.0
    v    = 75.0
    r    = 0.7
    lnA  = 2.5

    grad = dMdr(z, v, r, lnA, linea, lineb, model=None) 

    print(grad)
    
    print('\n\nDone.\n\n')
