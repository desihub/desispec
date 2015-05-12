"""
test desispec.fiberflat
"""

import unittest
import numpy as np
import scipy.sparse

from desispec.fiberflat import compute_fiberflat
from desispec.log import get_logger

#- Create a DESI logger at level WARNING to quiet down the fiberflat calc
import logging
log = get_logger(logging.WARNING)

def _get_data():
    """
    Return basic test data:
      - 1D wave[nwave]
      - 2D flux[nspec, nwave]
      - 2D ivar[nspec, nwave]
    """
    nspec = 10
    nwave = 100
    wave = np.linspace(0, np.pi, nwave)
    y = np.sin(wave)
    flux = np.tile(y, nspec).reshape(nspec, nwave)
    ivar = np.ones(flux.shape)
    
    return wave, flux, ivar
    

class TestFiberFlat(unittest.TestCase):

    def test_example(self):
        self.assertTrue( True )
        self.assertEqual(2+1, 4-1)
        self.assertAlmostEqual(1.0, 1.0+1e-12)
        
    def test_interface(self):
        """
        Basic test that interface works and identical inputs result in
        identical outputs
        """
        wave, flux, ivar = _get_data()
        nspec, nwave = flux.shape
        
        #- Setup data for a Resolution matrix
        sigma = 4.0
        ndiag = 21
        xx = np.linspace(-(ndiag-1)/2.0, +(ndiag-1)/2.0, ndiag)
        R = np.zeros( (nspec, ndiag, nwave) )
        for i in range(nspec):
            for j in range(nwave):
                kernel = np.exp(-xx**2/(2*sigma))
                kernel /= sum(kernel)
                R[i,:,j] = kernel

        #- Run the code
        fiberflat, ffivar, fmask, meanspec = compute_fiberflat(wave,flux,ivar,R)
            
        #- Check shape of outputs
        self.assertEqual(fiberflat.shape, flux.shape)
        self.assertEqual(ffivar.shape, flux.shape)
        self.assertEqual(fmask.shape, flux.shape)
        self.assertEqual(len(meanspec), nwave)
        
        #- Identical inputs should result in identical ouputs
        for i in range(1, nspec):
            self.assertTrue(np.all(fiberflat[i] == fiberflat[0]))
            self.assertTrue(np.all(ffivar[i] == ffivar[0]))
        
    def test_resolution(self):
        """
        Test that identical spectra convolved with different resolutions
        results in identical fiberflats
        """
        wave, flux, ivar = _get_data()
        nspec, nwave = flux.shape
        
        #- Setup a Resolution matrix that varies with fiber and wavelength
        #- Note: this is actually the transpose of the resolution matrix
        #- I wish I was creating, but as long as we self-consistently
        #- use it for convolving and solving, that shouldn't matter.
        sigma = np.linspace(2, 10, nwave*nspec)
        ndiag = 21
        xx = np.linspace(-ndiag/2.0, +ndiag/2.0, ndiag)
        R = np.zeros( (nspec, len(xx), nwave) )
        for i in range(nspec):
            for j in range(nwave):
                kernel = np.exp(-xx**2/(2*sigma[i*nwave+j]**2))
                kernel /= sum(kernel)
                R[i,:,j] = kernel

        #- Convolve the data with the resolution matrix
        offsets = range(ndiag//2, -ndiag//2, -1)
        convflux = np.empty_like(flux)
        for i in range(nspec):
            D = scipy.sparse.dia_matrix( (R[i], offsets), (nwave,nwave) )
            convflux[i] = D.dot(flux[i])

        #- Run the code
        fiberflat, ffivar, fmask, meanspec = compute_fiberflat(wave,convflux,ivar,R)

        #- These fiber flats should all be ~1
        self.assertTrue( np.all(np.abs(fiberflat-1) < 0.001) )

    #- Tests to implement.  Remove the "expectedFailure" line when ready.
    @unittest.expectedFailure
    def test_throughput(self):
        """
        Test that spectra with different throughputs but the same resolution
        result in the expected variations in fiberflat.
        """
        raise NotImplementedError

    @unittest.expectedFailure
    def test_throughput_resolution(self):
        """
        Test that spectra with different throughputs and different resolutions
        result in fiberflat variations that are only due to throughput.
        """
        raise NotImplementedError
        

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
