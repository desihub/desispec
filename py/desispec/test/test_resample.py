"""
tests desispec.interpolation.resample_flux
"""

import unittest, os
import numpy as np
from math import log

from desispec.interpolation import resample_flux, bin_bounds

class TestResample(unittest.TestCase):
    """
    Unit tests for interpolation.resample_flux
    """
        
    def test_resample(self):
        n = 100
        x = np.arange(n)
        y = np.ones(n)
        # we need in this test to make sure we have the same boundaries of the edges bins
        # to obtain the same flux density on the edges
        # because the resampling routine considers the flux is 0 outside of the input bins
        nout = n/2
        stepout = n/float(nout)
        xout = np.arange(nout)*stepout+stepout/2-0.5 
        yout = resample_flux(xout, x, y)
        self.assertTrue(np.all(yout == 1.0))                
    
    def test_non_uniform_grid(self):
        n = 100
        x = np.arange(n)+1.
        y = np.ones(n)
        # we need in this test to make sure we have the same boundaries of the edges bins
        # to obtain the same flux density on the edges
        # because the resampling routine considers the flux is 0 outside of the input bins
        # we consider here a logarithmic output grid
        nout = n/2
        lstepout = (log(x[-1])-log(x[0]))/float(nout)
        xout = np.exp(np.arange(nout)*lstepout)-0.5
        xout[0]  = x[0]-0.5+(xout[1]-xout[0])/2 # same edge of first bin
        offset   =  x[-1]+0.5-(xout[-1]-xout[-2])/2 - xout[-1]
        xout[-2:] += offset # same edge of last bin
        
        yout = resample_flux(xout, x, y)
        
        self.assertTrue(np.all(yout == 1.0))                
        
        
    def test_flux_conservation(self):
        n = 100
        x = np.arange(n)
        y = 1+np.sin(x/20.0)
        y[n/2+1] += 10
        # xout must have edges including bin half width equal
        # or larger than input to get the same integrated flux
        xout = np.arange(0,n+1,2)
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
            xout = np.arange(0,n+1,rebin)
            yout, ivout = resample_flux(xout, x, y, ivar)
            self.assertEqual(len(xout), len(yout))
            self.assertEqual(len(xout), len(ivout))
            # we have to compare the variance of ouput bins that
            # are fully contained in input
            self.assertAlmostEqual(ivout[ivout.size/2], ivar[ivar.size/2]*rebin)
            # check sum of weights is conserved 
            ivar_in  = np.sum(ivar)
            ivar_out = np.sum(ivout)
            self.assertAlmostEqual(ivar_in,ivar_out)
            
    def test_bin_bounds(self):
        """Super basic test only"""
        x = np.arange(10)
        lo, hi = bin_bounds(x)
        self.assertEqual(len(lo), len(x))
        self.assertEqual(len(hi), len(x))
        self.assertTrue(np.all(lo[1:] == hi[0:-1]))
        dx = x[1]-x[0]
        self.assertAlmostEqual(lo[0], x[0]-0.5*dx)
        self.assertAlmostEqual(lo[1], x[0]+0.5*dx)
        self.assertAlmostEqual(hi[-1], x[-1]+0.5*dx)
            

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
