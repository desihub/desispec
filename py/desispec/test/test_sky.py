"""
tests desispec.sky
"""

import unittest

import numpy as np
from desispec.sky import compute_sky, subtract_sky
from desispec.resolution import Resolution
from desispec.frame import Frame
import desispec.io

class TestSky(unittest.TestCase):
    
    #- Create unique test filename in a subdirectory
    def setUp(self):
        #- Create a fake sky
        self.nspec = 10
        self.wave = np.arange(4000, 4500)
        self.nwave = len(self.wave)
        self.flux = np.zeros(self.nwave)
        for i in range(0, self.nwave, 20):
            self.flux[i] = i
            
        self.ivar = np.ones(self.flux.shape)
                    
    def _get_spectra(self):
        #- Setup data for a Resolution matrix
        sigma = 4.0
        ndiag = 21
        xx = np.linspace(-(ndiag-1)/2.0, +(ndiag-1)/2.0, ndiag)
        Rdata = np.zeros( (self.nspec, ndiag, self.nwave) )
        for i in range(self.nspec):
            for j in range(self.nwave):
                kernel = np.exp(-xx**2/(2*sigma))
                kernel /= sum(kernel)
                Rdata[i,:,j] = kernel
                
        flux = np.zeros((self.nspec, self.nwave))
        ivar = np.ones((self.nspec, self.nwave))
        mask = np.zeros((self.nspec, self.nwave), dtype=int)
        for i in range(self.nspec):
            R = Resolution(Rdata[i])
            flux[i] = R.dot(self.flux)

        fibermap = desispec.io.empty_fibermap(self.nspec)
        fibermap['OBJTYPE'][0::2] = 'SKY'

        return Frame(self.wave, flux, ivar, mask, Rdata, spectrograph=0), fibermap
                    
    def test_uniform_resolution(self):        
        #- Setup data for a Resolution matrix
        spectra, fibermap = self._get_spectra()
                        
        sky = compute_sky(spectra, fibermap)
        self.assertEqual(sky.flux.shape, spectra.flux.shape)
        self.assertEqual(sky.ivar.shape, spectra.ivar.shape)
        self.assertEqual(sky.mask.shape, spectra.mask.shape)
        
        delta=spectra.flux[0]-sky.flux[0]
        d=np.inner(delta,delta)
        self.assertAlmostEqual(d,0.)
        
        delta=spectra.flux[-1]-sky.flux[-1]
        d=np.inner(delta,delta)
        self.assertAlmostEqual(d,0.)

    def test_subtract_sky(self):
        spectra, fibermap = self._get_spectra()
        sky = compute_sky(spectra, fibermap)
        subtract_sky(spectra, sky)
        #- allow some slop in the sky subtraction
        self.assertTrue(np.allclose(spectra.flux, 0, rtol=1e-5, atol=1e-6))
        
    def runTest(self):
        pass
                
if __name__ == '__main__':
    unittest.main()
