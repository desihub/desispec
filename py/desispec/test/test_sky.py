"""
tests desispec.sky
"""

import unittest

import numpy as np
from desispec.sky import compute_sky
from desispec.resolution import Resolution

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
                    
    def test_uniform_resolution(self):        
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
        for i in range(self.nspec):
            R = Resolution(Rdata[i])
            flux[i] = R.dot(self.flux)

                        
        skyflux, skyivar, skymask, cskyflux, cskyivar = compute_sky(self.wave, flux, ivar, Rdata)
        self.assertEqual(len(skyflux.shape), 1)
        self.assertEqual(len(skyflux), self.nwave)
        self.assertEqual(len(skyflux), len(skyivar))
        self.assertEqual(len(skyflux), len(skymask))
        delta=flux[0]-cskyflux
        d=np.inner(delta,delta)
        self.assertAlmostEqual(d,0.)
        delta=flux[-1]-cskyflux
        d=np.inner(delta,delta)
        self.assertAlmostEqual(d,0.)
        


    def runTest(self):
        pass
                
if __name__ == '__main__':
    unittest.main()
