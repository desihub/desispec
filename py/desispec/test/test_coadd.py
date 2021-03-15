import unittest

import numpy as np
from desispec.spectra import Spectra
from desispec.io import empty_fibermap
from desispec.coaddition import coadd,fast_resample_spectra,spectroperf_resample_spectra
from desispec.maskbits import fibermask

class TestCoadd(unittest.TestCase):
        
    def _random_spectra(self, ns=3, nw=10):
        
        wave = np.linspace(5000, 5100, nw)
        flux = np.random.uniform(0, 1, size=(ns,nw))
        ivar = np.random.uniform(0, 1, size=(ns,nw))
        #mask = np.zeros((ns,nw),dtype=int)
        mask = None
        rdat = np.ones((ns,3,nw))
        rdat[:,0] *= 0.25
        rdat[:,1] *= 0.5
        rdat[:,2] *= 0.25
        fmap = empty_fibermap(ns)
        fmap["TARGETID"][:]=12 # same target
        return Spectra(bands=["x"],wave={"x":wave},flux={"x":flux},ivar={"x":ivar}, mask=None, resolution_data={"x":rdat} , fibermap=fmap)
        

        
    def test_coadd(self):
        """Test coaddition"""
        s1 = self._random_spectra(3,10)
        coadd(s1)
        
    def test_spectroperf_resample(self):
        """Test spectroperf_resample"""
        s1 = self._random_spectra(1,20)
        wave = np.linspace(5000, 5100, 10)
        s2 = spectroperf_resample_spectra(s1,wave=wave)
        
    def test_fast_resample(self):
        """Test fast_resample"""
        s1 = self._random_spectra(1,20)
        wave = np.linspace(5000, 5100, 10)
        s2 = fast_resample_spectra(s1,wave=wave)

    def test_fiberstatus(self):
        """Test that FIBERSTATUS=0 isn't included in coadd"""
        def _makespec(nspec, nwave):
            s1 = self._random_spectra(nspec, nwave)
            s1.flux['x'][:,:] = 1.0
            s1.ivar['x'][:,:] = 1.0
            return s1

        #- Nothing masked
        nspec, nwave = 4,10
        s1 = _makespec(nspec, nwave)
        expt = 33 # random number
        s1.fibermap['EXPTIME'][:]=expt
        self.assertEqual(len(s1.fibermap), nspec)
        coadd(s1)
        self.assertEqual(len(s1.fibermap), 1)
        self.assertEqual(s1.fibermap['COADD_NUMEXP'][0], nspec)
        self.assertEqual(s1.fibermap['COADD_EXPTIME'][0], expt*nspec)
        self.assertEqual(s1.fibermap['FIBERSTATUS'][0], 0)
        self.assertTrue(np.all(s1.flux['x'] == 1.0))
        self.assertTrue(np.allclose(s1.ivar['x'], 1.0*nspec))

        #- Two spectra masked
        nspec, nwave = 5,10
        s1 = _makespec(nspec, nwave)
        self.assertEqual(len(s1.fibermap), nspec)

        s1.fibermap['FIBERSTATUS'][0] = fibermask.BROKENFIBER
        s1.fibermap['FIBERSTATUS'][1] = fibermask.BADFIBER
 
        coadd(s1)
        self.assertEqual(len(s1.fibermap), 1)
        self.assertEqual(s1.fibermap['COADD_NUMEXP'][0], nspec-2)
        self.assertEqual(s1.fibermap['FIBERSTATUS'][0], 0)
        self.assertTrue(np.all(s1.flux['x'] == 1.0))
        self.assertTrue(np.allclose(s1.ivar['x'], 1.0*(nspec-2)))

        #- All spectra masked
        nspec, nwave = 5,10
        s1 = _makespec(nspec, nwave)
        self.assertEqual(len(s1.fibermap), nspec)

        s1.fibermap['FIBERSTATUS'] = fibermask.BROKENFIBER
        
        coadd(s1)
        self.assertEqual(len(s1.fibermap), 1)
        self.assertEqual(s1.fibermap['COADD_NUMEXP'][0], 0)
        self.assertEqual(s1.fibermap['FIBERSTATUS'][0], fibermask.BROKENFIBER)
        self.assertTrue(np.all(s1.flux['x'] == 0.0))
        self.assertTrue(np.all(s1.ivar['x'] == 0.0))

if __name__ == '__main__':
    unittest.main()           
