"""
tests desispec.interpolation.resample_flux
"""

import unittest, os
import numpy as np
from math import log

from desispec.interpolation import resample_flux

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
        nout = n//2
        stepout = n/float(nout)
        xout = np.arange(nout)*stepout+stepout/2-0.5 
        yout = resample_flux(xout, x, y, extrapolate=True)
        self.assertTrue(np.all(yout == 1.0))                

    def test_non_uniform_grid(self):
        n = 100
        x = np.arange(n)+1.
        y = np.ones(n)
        # we need in this test to make sure we have the same boundaries of the edges bins
        # to obtain the same flux density on the edges
        # because the resampling routine considers the flux is 0 outside of the input bins
        # we consider here a logarithmic output grid
        nout = n//2
        lstepout = (log(x[-1])-log(x[0]))/float(nout)
        xout = np.exp(np.arange(nout)*lstepout)-0.5
        xout[0]  = x[0]-0.5+(xout[1]-xout[0])/2 # same edge of first bin
        offset   =  x[-1]+0.5-(xout[-1]-xout[-2])/2 - xout[-1]
        xout[-2:] += offset # same edge of last bin
        
        yout = resample_flux(xout, x, y,extrapolate=True)
        zero = np.max(np.abs(yout-1))
        self.assertAlmostEqual(zero,0.)

    def test_flux_conservation(self):
        n = 100
        x = np.arange(n)
        #y = 1+np.sin(x/20.0)
        y = 0.4*x+10 # only exact for linear relation
        y[n//2+1] += 10
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
        y[n//2+1] += 10
        ivar = np.ones(n)
        for rebin in (2, 3, 5):
            xout = np.arange(0,n+1,rebin)
            yout, ivout = resample_flux(xout, x, y, ivar)
            self.assertEqual(len(xout), len(yout))
            self.assertEqual(len(xout), len(ivout))
            # we have to compare the variance of ouput bins that
            # are fully contained in input
            self.assertAlmostEqual(ivout[ivout.size//2], ivar[ivar.size//2]*rebin)
            # check sum of weights is conserved 
            ivar_in  = np.sum(ivar)
            ivar_out = np.sum(ivout)
            self.assertAlmostEqual(ivar_in,ivar_out)


    # def test_same_bin(self):
    #     '''test reproducibility if two input bins are the same'''
    #     x  = np.array([1, 2, 3, 3, 4, 5])
    #     y1 = np.array([1, 2, 3, 4, 5, 6])
    #     y2 = np.array([1, 2, 4, 3, 5, 6])
    #     xx = np.array([1, 2.5, 3.3, 4.5])
    #     z1 = resample_flux(xx, x, y1)
    #     z2 = resample_flux(xx, x, y2)
    #     self.assertTrue(np.all(z1 == z2))

    @unittest.expectedFailure
    def test_edges(self):
        '''Test for large edge effects in resampling'''
        x = np.arange(0.0, 100)
        y = np.sin(x/20)
        xx = np.linspace(1, 99, 23)
        yy = resample_flux(xx, x, y)
        diff = np.abs(yy - np.interp(xx, x, y))
        self.assertLess(np.max(np.abs(diff)), 1e-2)
