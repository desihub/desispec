import unittest

import numpy as np
from spectra import Spectra
from desispec.resolution import Resolution

class TestSpectra(unittest.TestCase):

    def test_init(self):        
        nspec = 3
        nwave = 10
        wave = np.arange(nwave)
        flux = np.random.uniform(size=(nspec, nwave))
        ivar = np.ones(flux.shape)
        rdata = np.ones((nspec, 5, nwave))

        sp = Spectra(wave, flux, ivar, rdata)
        self.assertTrue(np.all(sp.wave == wave))
        self.assertTrue(np.all(sp.flux == flux))
        self.assertTrue(np.all(sp.ivar == ivar))
        self.assertTrue(np.all(sp.resolution_data == rdata))
        self.assertEqual(sp.nspec, nspec)
        self.assertEqual(sp.nwave, nwave)
        self.assertTrue(isinstance(sp.R[0], Resolution))
        #- check dimensionality mismatches
        self.assertRaises(ValueError, lambda x: Spectra(*x), (wave, wave, ivar, rdata))
        self.assertRaises(ValueError, lambda x: Spectra(*x), (wave, flux[0:2], ivar, rdata))

    def test_slice(self):
        nspec = 5
        nwave = 10
        wave = np.arange(nwave)
        flux = np.random.uniform(size=(nspec, nwave))
        ivar = np.ones(flux.shape)
        rdata = np.ones((nspec, 5, nwave))

        sp = Spectra(wave, flux, ivar, rdata)
        x = sp[1]
        x = sp[1:2]
        x = sp[[1,2,3]]
        x = sp[sp.fibers<3]

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
