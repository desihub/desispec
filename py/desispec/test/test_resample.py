import unittest, os
import numpy as np

from desispec.interpolation import resample_flux

class TestResample(unittest.TestCase):
    """
    Unit tests for interpolation.resample_flux
    """
        
    def test_resample(self):
        n = 100
        x = np.arange(n)
        y = np.ones(n)
        xout = np.arange(0,n,2)
        yout = resample_flux(xout, x, y)
        self.assertTrue(np.all(yout == 1.0))                
        
    ### @unittest.expectedFailure
    def test_flux_conservation(self):
        n = 100
        x = np.arange(n)
        y = 1+np.sin(x/20.0)
        y[n/2+1] += 10
        xout = np.arange(0,n,2)
        yout = resample_flux(xout, x, y)
        
        fluxin = np.sum(y*np.gradient(x))
        fluxout = np.sum(yout*np.gradient(xout))
        self.assertAlmostEqual(fluxin, fluxout)

    #- Confirm weights increase by X when rebinning by X
    def test_weighted_resample(self):
        n = 100
        x = np.arange(n)
        y = 1+np.sin(x/20.0)        
        y[n/2+1] += 10
        ivar = np.ones(n)
        for rebin in (2, 3, 5):
            xout = np.arange(0,n,rebin)
            yout, ivout = resample_flux(xout, x, y, ivar)
            self.assertEqual(len(xout), len(yout))
            self.assertEqual(len(xout), len(ivout))
            self.assertAlmostEqual(ivout[0], ivar[0]*rebin)

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
