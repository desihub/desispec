"""
test desispec.fiberflat
"""

import unittest
import numpy as np
import scipy.sparse

from desispec.resolution import Resolution
from desispec.frame import Frame
from desispec.fiberflat import FiberFlat
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
    mask = np.zeros(flux.shape, dtype=int)
    
    return wave, flux, ivar, mask
    

class TestFiberFlat(unittest.TestCase):

    def test_interface(self):
        """
        Basic test that interface works and identical inputs result in
        identical outputs
        """
        wave, flux, ivar, mask = _get_data()
        nspec, nwave = flux.shape
        
        #- Setup data for a Resolution matrix
        sigma = 4.0
        ndiag = 21
        xx = np.linspace(-(ndiag-1)/2.0, +(ndiag-1)/2.0, ndiag)
        Rdata = np.zeros( (nspec, ndiag, nwave) )
        for i in range(nspec):
            for j in range(nwave):
                kernel = np.exp(-xx**2/(2*sigma))
                kernel /= sum(kernel)
                Rdata[i,:,j] = kernel

        #- Run the code
        frame = Frame(wave, flux, ivar, mask, Rdata, spectrograph=0)
        ff = compute_fiberflat(frame)
            
        #- Check shape of outputs
        self.assertEqual(ff.fiberflat.shape, flux.shape)
        self.assertEqual(ff.ivar.shape, flux.shape)
        self.assertEqual(ff.mask.shape, flux.shape)
        self.assertEqual(len(ff.meanspec), nwave)
        
        #- Identical inputs should result in identical ouputs
        for i in range(1, nspec):
            self.assertTrue(np.all(ff.fiberflat[i] == ff.fiberflat[0]))
            self.assertTrue(np.all(ff.ivar[i] == ff.ivar[0]))
        
    def test_resolution(self):
        """
        Test that identical spectra convolved with different resolutions
        results in identical fiberflats
        """
        wave, flux, ivar, mask = _get_data()
        nspec, nwave = flux.shape
        
        #- Setup a Resolution matrix that varies with fiber and wavelength
        #- Note: this is actually the transpose of the resolution matrix
        #- I wish I was creating, but as long as we self-consistently
        #- use it for convolving and solving, that shouldn't matter.
        sigma = np.linspace(2, 10, nwave*nspec)
        ndiag = 21
        xx = np.linspace(-ndiag/2.0, +ndiag/2.0, ndiag)
        Rdata = np.zeros( (nspec, len(xx), nwave) )
        for i in range(nspec):
            for j in range(nwave):
                kernel = np.exp(-xx**2/(2*sigma[i*nwave+j]**2))
                kernel /= sum(kernel)
                Rdata[i,:,j] = kernel

        #- Convolve the data with the resolution matrix
        convflux = np.empty_like(flux)
        for i in range(nspec):
            convflux[i] = Resolution(Rdata[i]).dot(flux[i])

        #- Run the code
        frame = Frame(wave, convflux, ivar, mask, Rdata, spectrograph=0)
        ff = compute_fiberflat(frame)

        #- These fiber flats should all be ~1
        self.assertTrue( np.all(np.abs(ff.fiberflat-1) < 0.001) )

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
        
class TestFiberFlatObject(unittest.TestCase):

    def setUp(self):
        self.nspec = 5
        self.nwave = 10
        self.wave = np.arange(self.nwave)
        self.fiberflat = np.random.uniform(size=(self.nspec, self.nwave))
        self.ivar = np.ones(self.fiberflat.shape)
        self.mask = np.zeros(self.fiberflat.shape)
        self.meanspec = np.random.uniform(size=self.nwave)
        self.ff = FiberFlat(self.wave, self.fiberflat, self.ivar, self.mask, self.meanspec)

    def test_init(self):
        for key in ('wave', 'fiberflat', 'ivar', 'mask', 'meanspec'):
            x = self.ff.__getattribute__(key)
            y = self.__getattribute__(key)
            self.assertTrue(np.all(x == y), key)

        self.assertEqual(self.nspec, self.ff.nspec)
        self.assertEqual(self.nwave, self.ff.nwave)

    def test_dimensions(self):
        #- check dimensionality mismatches
        self.assertRaises(ValueError, lambda x: FiberFlat(*x), (self.wave, self.wave, self.ivar, self.mask, self.meanspec))
        self.assertRaises(ValueError, lambda x: FiberFlat(*x), (self.wave, self.fiberflat, self.ivar, self.mask, self.fiberflat))
        self.assertRaises(ValueError, lambda x: FiberFlat(*x), (self.wave, self.fiberflat[0:2], self.ivar, self.mask, self.meanspec))

    def test_slice(self):
        x = self.ff[1]
        x = self.ff[1:2]
        x = self.ff[[1,2,3]]
        x = self.ff[self.ff.fibers<3]


#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
