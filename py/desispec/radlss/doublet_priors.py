import numpy as np


def jeffreys_prior(x, sigma):
    # unnormalised.  Applied to line flux, line ratio & velocity dispersion, both positive definite.  
    return  1. / sigma

def gaussian_prior(x, mu, sigma):
    # unnormalised (1. / (np.sqrt(2. * np.pi) * sigma)).  Applied to redrock z; NOTE: rrzerr ~ 1.e-5 so normaliation
    # drives to greater than unity.
    return  np.exp(-0.5 * (x - mu)**2. / sigma**2.)

def doublet_prior(z, v, r, lnA, rrz, rrzerr):
    prior = gaussian_prior(z, rrz, rrzerr) * gaussian_prior(v, 91, 50.) * gaussian_prior(r, 0.71, 0.1)

    return  prior
    
def mlogprior(p):
    # prior on the parameter space.    
    with np.errstate(divide='raise'):
      try:
        return  -np.log(p)
    
      except FloatingPointError:
        # Divide by zero caused by log(0.0);
        return  1.e99

if __name__ == '__main__':
    # 1.0157650709152222, 3.1128353189916926e-05, [1.01575065, 58.94169126, -0.98103215, -22.52085698, 3.92917686, 6.37407076, 0.65365138] -9.35107983777596
    rrz    = 1.01575065
    rrzerr = 3.1128353189916926e-05

    z      = 1.01575065
    v      = 58.94169126
    
    prior  = gaussian_prior(z, rrz, rrzerr)
    m      = mlogprior(prior)

    print(prior, m, np.sqrt(2. * m))

    print('\n\nDone.\n\n')
