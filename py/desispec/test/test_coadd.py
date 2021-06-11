import unittest

import numpy as np
from astropy.table import Table
from desispec.spectra import Spectra
from desispec.io import empty_fibermap
from desispec.coaddition import (coadd, fast_resample_spectra,
        spectroperf_resample_spectra, coadd_fibermap)

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
        

        
    def _test_coadd(self):
        """Test coaddition"""
        s1 = self._random_spectra(3,10)
        coadd(s1)
        
    def _test_spectroperf_resample(self):
        """Test spectroperf_resample"""
        s1 = self._random_spectra(1,20)
        wave = np.linspace(5000, 5100, 10)
        s2 = spectroperf_resample_spectra(s1,wave=wave)
        
    def _test_fast_resample(self):
        """Test fast_resample"""
        s1 = self._random_spectra(1,20)
        wave = np.linspace(5000, 5100, 10)
        s2 = fast_resample_spectra(s1,wave=wave)

    def test_coadd_fibermap_onetile(self):
        """Test coadding a fibermap of a single tile"""
        #- one tile, 3 targets, 2 exposures on 2 nights
        fm = Table()
        fm['TARGETID'] = [111,111,222,222,333,333]
        fm['DESI_TARGET'] = [4,4,8,8,16,16]
        fm['TILEID'] = [1,1,1,1,1,1]
        fm['NIGHT'] = [20201220,20201221]*3
        fm['EXPID'] = [10,20,11,21,12,22]
        fm['FIBER'] = [5,6,]*3
        fm['FIBERSTATUS'] = [0,0,0,0,0,0]
        fm['FIBER_X'] = [1.1, 2.1]*3
        fm['FIBER_Y'] = [10.2, 5.3]*3
        fm['FLUX_R'] = np.ones(6)

        cofm, expfm = coadd_fibermap(fm, onetile=True)

        #- Single tile coadds include these in the coadded fibermap
        for col in ['TARGETID', 'DESI_TARGET',
                'TILEID', 'FIBER', 'FIBERSTATUS', 'FLUX_R',
                'MEAN_FIBER_X', 'MEAN_FIBER_Y']:
            self.assertIn(col, cofm.colnames)

        #- but these columns should not be in the coadd
        for col in ['NIGHT', 'EXPID', 'FIBER_X', 'FIBER_Y']:
            self.assertNotIn(col, cofm.colnames)

        #- the exposure-level fibermap has columns specific to individual
        #- exposures, but drops some of the target-level columns
        for col in ['TARGETID', 'TILEID', 'NIGHT', 'EXPID', 'FIBER',
                'FIBERSTATUS', 'FIBER_X', 'FIBER_Y']:
            self.assertIn(col, expfm.colnames)

        for col in ['DESI_TARGET', 'FLUX_R']:
            self.assertNotIn(col, expfm.colnames)

        #- onetile coadds should fail if input has multiple tiles
        fm['TILEID'][0] += 1
        with self.assertRaises(ValueError):
            cofm, expfm = coadd_fibermap(fm, onetile=True)


    def test_coadd_fibermap_multitile(self):
        """Test coadding a fibermap covering multiple tiles"""
        #- Target 111 observed on tile 1 on one night
        #- Target 222 observed on tiles 1 and 2 on two nights
        #- Target 333 observed on tile 2 on one night
        fm = Table()
        fm['TARGETID'] = [111,111,222,222,333,333]
        fm['DESI_TARGET'] = [4,4,8,8,16,16]
        fm['TILEID'] = [1,1,1,2,2,2]
        fm['NIGHT'] = [20201220,]*3 + [20201221,]*3
        fm['EXPID'] = [100,]*3 + [101,]*3
        fm['FIBER'] = [5,5,6,7,8,8]
        fm['FIBERSTATUS'] = [0,0,0,0,0,0]
        fm['FIBER_X'] = [1.1, 1.1, 2.2, 5.6, 10.2, 10.1]
        fm['FIBER_Y'] = [2.2, 2.2, 10.3, -0.1, 0.2, 0.3]
        fm['FLUX_R'] = np.ones(6)

        cofm, expfm = coadd_fibermap(fm, onetile=False)

        #- Multi tile coadds include these in the coadded fibermap
        for col in ['TARGETID', 'DESI_TARGET', 'FIBERSTATUS',
                'FIBERSTATUS', 'FLUX_R']:
            self.assertIn(col, cofm.colnames)

        #- but these columns should not be in the coadd
        for col in ['NIGHT', 'EXPID', 'TILEID', 'FIBER',
                'FIBER_X', 'FIBER_Y', 'MEAN_FIBER_X', 'MEAN_FIBER_Y']:
            self.assertNotIn(col, cofm.colnames)

        #- the exposure-level fibermap has columns specific to individual
        #- exposures, but drops some of the target-level columns
        for col in ['TARGETID', 'TILEID', 'NIGHT', 'EXPID', 'FIBER',
                'FIBERSTATUS', 'FIBER_X', 'FIBER_Y']:
            self.assertIn(col, expfm.colnames)

        for col in ['DESI_TARGET', 'FLUX_R']:
            self.assertNotIn(col, expfm.colnames)


    def _test_fiberstatus(self):
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
