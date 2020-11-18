import numpy as np


def jeffreys_prior(x, sigma):
    # unnormalised.  Applied to line flux, line ratio & velocity dispersion, both positive definite.  
    return  1. / sigma

def gaussian_prior(x, mu, sigma):
    # normalised.  Applied to redrock z.
    return  (1. / np.sqrt(2. * np.pi) / sigma) * np.exp(-0.5 * (x - mu)**2. / sigma**2.)

def doublet_prior(z, v, r, lnA, rrz, rrzerr):
    # Gaussian prior in redshift.
    muz  = rrz     
    sigz = rrzerr

    # Jeffreys prior in line flux, dispersion, line ratio. 
    sigv = 50.
    sigr =  1.
    sigA =  5. 
    
    # unnormalised (scalar).
    return  gaussian_prior(z, muz, sigz) * jeffreys_prior(v, sigv) * gaussian_prior(r, 0.7, 0.1) # * gaussian_prior(lnA, 1.0, 1.0)

def mlogprior(x, rrz, rrzerr):
    z     = x[0]
    v     = x[1]
    r     = x[2]
    lnA   = x[3]

    return  0.5 * (z - rrz)**2. / rrzerr**2.
    
    '''
    # prior on the parameter space.
    p     = doublet_prior(z, v, r, lnA, rrz, rrzerr)
    
    with np.errstate(divide='raise'):
      try:
        return  -np.log(p)
    
      except FloatingPointError:
        # Divide by zero caused by log(0.0);
        return  1.e99
    '''

if __name__ == '__main__':
    x = jeffreys_prior(1.0, 2.0)
    y = gaussian_prior(1.0, 1.1, 0.2)
    z = doublet_prior(1., 200., 0.7, 0.0)
    m = mlogprior(np.array([1., 200., 0.7, 0.0]))

    for p in [x, y, z, m]:
        print(p)

    print('\n\nDone.\n\n')
