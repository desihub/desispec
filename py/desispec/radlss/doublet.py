import numpy as np
import astropy.constants as const

from   lines import lines


def doublet(z, sigmav=10., r=0.1, lineida=6, lineidb=7, _twave=None):
      '''
      See https://arxiv.org/pdf/2007.14484.pdf
      
      sigma:  velocity term, later convolved for resolution. 
      '''  

      if _twave is None:
          _twave = np.arange(3100., 10400., 0.1)

      linea      = lines.loc[lineida, 'WAVELENGTH']
      lineb      = lines.loc[lineidb, 'WAVELENGTH']
      
      lightspeed = const.c.to('km/s').value
      sigma_lam  = sigmav * (1. + z) * linea / lightspeed    
        
      # Line flux of 1 erg/s/cm2/Angstrom, sigma is the width of the line, z is the redshift and r is the relative amplitudes of the lines in the doublet. 
      return  _twave, 1. / (1. + r) / np.sqrt(2. * np.pi) / sigma_lam * (r * np.exp(- ((_twave - linea * (1. + z)) / np.sqrt(2.) / sigma_lam)**2.) + np.exp(- ((_twave - lineb * (1. + z)) / np.sqrt(2.) / sigma_lam)**2.))


if __name__ == '__main__':
      import pylab as pl

      
      redshift   = 1.00
      wave, flux = doublet(z=redshift, sigmav=10., r=0.0, lineida=6, lineidb=7, _twave=None)

      oii        = (1. + redshift) * lines.loc[6, 'WAVELENGTH']
      
      pl.plot(wave, flux)

      pl.xlim(oii - 25., oii + 25.)

      pl.show()
      
      print('\n\nDone.\n\n')
