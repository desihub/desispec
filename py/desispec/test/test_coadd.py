import unittest

import numpy as np
from desispec.spectra import Spectra
from desispec.io import empty_fibermap
from desispec.coaddition import coadd,fast_resample_spectra,spectroperf_resample_spectra

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
        

if __name__ == '__main__':
    unittest.main()           
