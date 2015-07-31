import unittest

import numpy as np
from desispec.coaddition import Spectrum
from desispec.resolution import Resolution

class TestCoadd(unittest.TestCase):
        
    def _getdata(self, n=10):
        wave = np.linspace(5000, 5100, n)
        flux = np.random.uniform(0, 1, size=n)
        ivar = np.random.uniform(0, 1, size=n)
        ### mask = np.random.randint(0, 256, size=n)
        rdat = np.ones((3, n))
        rdat[0] *= 0.25
        rdat[1] *= 0.5
        rdat[2] *= 0.25
        R = Resolution(rdat)
        ### return wave, flux, ivar, mask, R
        return wave, flux, ivar, None, R
        
    def test_spectrum(self):
        """Test basic constructor interface"""
        wave, flux, ivar, mask, R = self._getdata(10)
        
        #- Each of these should be allowable
        s = Spectrum(wave)
        s = Spectrum(wave, flux)
        s = Spectrum(wave, flux, ivar, mask, R)

        #- But ivar does require R
        self.assertRaises(AssertionError, lambda x: Spectrum(*x), (wave, flux, ivar))
        self.assertRaises(AssertionError, lambda x: Spectrum(*x), (wave, flux, ivar, mask))

        #- did it get filled in?
        self.assertTrue(np.array_equal(s.wave, wave))
        self.assertTrue(np.array_equal(s.flux, flux))
        self.assertTrue(np.array_equal(s.ivar, ivar))
        
    def test_basic_coadd(self):
        """Test coaddition on a common wavelength grid"""
        n = 10
        s1 = Spectrum(*self._getdata(n))
        s1 += Spectrum(*self._getdata(n))
        s1 += Spectrum(*self._getdata(n))
        
        self.assertEqual(s1.flux, None)
        s1.finalize()
        self.assertTrue(s1.flux is not None)
        self.assertTrue(s1.flux.shape == flux.shape)
        
    def test_nonuniform_coadd(self):
        """Test coaddition of spectra with different wavelength grids"""
        s1 = Spectrum(*self._getdata(10))
        s1 += Spectrum(*self._getdata(13))

if __name__ == '__main__':
    unittest.main()           
