import time
import numba
import numpy             as np 
import astropy.constants as const

from   lines             import lines
from   numba             import jit
from   twave             import twave


lightspeed = const.c.to('km/s').value

def sig_lambda(z, sigmav, lineb):
      return  sigmav * (1. + z) * lineb / lightspeed

@jit(nopython=True)
def doublet(z, twave, sigmav=10., r=0.1, linea=3726.032, lineb=3728.815):
      '''
      See https://arxiv.org/pdf/2007.14484.pdf
      
      sigma:  velocity term, later convolved for resolution. 
      '''  

      # sig_lambda(z, sigmav, lineb)
      sigma_lam    = sigmav * (1. + z) * lineb / lightspeed
      
      # Line flux of 1 erg/s/cm2/Angstrom, sigma is the width of the line, z is the redshift and r is the relative amplitudes of the lines in the doublet. 
      result       = (1. / (1. + r) / np.sqrt(2. * np.pi) / sigma_lam / sigma_lam) * (r * np.exp(- ((twave - linea * (1. + z)) / np.sqrt(2.) / sigma_lam)**2. ) + np.exp(- ((twave - lineb * (1. + z)) / np.sqrt(2.) / sigma_lam)**2.))
      
      # print(z, sigmav, r, linea, lineb, np.sum(result))      
      return  twave, result


if __name__ == '__main__':
      import pylab as pl

      
      redshift   = 1.00
      
      lineida    = 6
      lineidb    = 7
      
      linea      = lines['WAVELENGTH'][lineida]
      lineb      = lines['WAVELENGTH'][lineidb]

      
      start = time.time()
      
      wave, flux = doublet(z=redshift, twave=twave, sigmav=10., r=0.0, linea=linea, lineb=lineb)

      end = time.time()

      print("Elapsed (with compilation) = %s" % (end - start))

      start = time.time()

      wave, flux = doublet(z=redshift, twave=twave, sigmav=10., r=0.0, linea=linea, lineb=lineb)

      end = time.time()

      print("Elapsed (after compilation) = %s" % (end - start))
      
      oii        = (1. + redshift) * lines['WAVELENGTH'][6]
      
      # pl.plot(wave, flux)

      # pl.xlim(oii - 25., oii + 25.)

      # pl.show()
      
      print('\n\nDone.\n\n')
