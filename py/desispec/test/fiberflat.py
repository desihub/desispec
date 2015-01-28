import unittest
import numpy as np

from desispec.fiberflat import compute_fiberflat

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
        nspec = 10
        nwave = 100
        
        #- Input wave, flux, ivar
        wave = np.linspace(0, np.pi, nwave)
        y = np.sin(wave)
        flux = np.tile(y, nspec).reshape(nspec, nwave)
        ivar = np.ones(flux.shape)
        
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
        fiberflat, ffivar, meanspec = compute_fiberflat(wave,flux,ivar,R)
            
        #- Check shape of outputs
        self.assertEqual(fiberflat.shape, flux.shape)
        self.assertEqual(ffivar.shape, flux.shape)
        self.assertEqual(len(meanspec), nwave)
        
        #- Identical inputs should result in identical ouputs
        for i in range(1, nspec):
            self.assertTrue(np.all(fiberflat[i] == fiberflat[0]))
            self.assertTrue(np.all(ffivar[i] == ffivar[0]))
        
    #- Tests to implement.  Remove the "expectedFailure" line when ready.
    @unittest.expectedFailure
    def test_resolution(self):
        """
        Test that identical spectra convolved with different resolutions
        results in identical fiberflats
        """
        raise NotImplementedError

    @unittest.expectedFailure
    def test_throughput(self):
        """
        Test that spectra with different throughputs but the same resolution
        result in the expected variations in fiberflat.
        """
        raise NotImplementedError
        

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
