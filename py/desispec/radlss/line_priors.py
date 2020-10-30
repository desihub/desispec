def jeffreys_prior(x, sigma):
    # unnormalised.  Applied to line flux, line ratio & velocity dispersion, both positive definite.  
    return  1. / sigma

def gaussian_prior(x, mu, sigma):
    # normalised.  Applied to redrock z.
    return  (1. / np.sqrt(2. * np.pi) / sigma) * np.exp(-0.5 * (x - mu)**2. / sigma**2.)

def line_prior(z, v, r, lnA):
    # Gaussian prior in redshift.
    muz  = zz    # rr bestfit z.
    sigz = zerr  # rr sigma z

    # Jeffreys prior in line flux, dispersion, line ratio. 
    sigv = 50.
    sigr =  1.
    sigA =  5. 
    
    # unnormalised (scalar).
    return  gaussian_prior(z, muz, sigz) * jeffreys_prior(v, sigv) * gaussian_prior(r, 0.7, 0.25) * gaussian_prior(lnA, 1.0, 1.0)
