import unittest

import numpy as np

from astropy.table import Table

from desispec.spectra import Spectra
from desispec.io import empty_fibermap
from desispec.coaddition import (coadd, fast_resample_spectra,
        spectroperf_resample_spectra, coadd_fibermap, coadd_cameras)
from desispec.specscore import compute_coadd_scores

from desispec.maskbits import fibermask

class TestCoadd(unittest.TestCase):

    def setUp(self):
        # Create a dummy 3-camera spectrum.
        bands = ['b', 'r', 'z']
        wave = { 'b':np.arange(3600, 5800.8, 0.8),
                 'r':np.arange(5760, 7620.8, 0.8),
                 'z':np.arange(7520, 9824.8, 0.8) }
        flux = { }
        ivar = { }
        rdat = { }
        for b in bands:
            flux[b] = np.ones((1, len(wave[b])))
            ivar[b] = np.ones((1, len(wave[b])))
            rdat[b] = np.ones((1, 1, len(wave[b])))

        fmap = empty_fibermap(1)
        fmap['TARGETID'] = 12
        fmap['TILEID'] = 1000
        fmap['NIGHT'] = 20200101
        fmap['EXPID'] = 5000 + np.arange(1)

        self.spectra = Spectra(bands=bands,
                               wave=wave,
                               flux=flux,
                               ivar=ivar,
                               mask=None,
                               resolution_data=rdat,
                               fibermap=fmap)
        
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
        #- add fibermap columns in spectra from Frame headers
        fmap["TARGETID"] = 12
        fmap["TILEID"] = 1000
        fmap["NIGHT"] = 20200101
        fmap["EXPID"] = 5000 + np.arange(ns)
        return Spectra(
                bands=["b"],
                wave={"b":wave},
                flux={"b":flux},
                ivar={"b":ivar},
                mask=None,
                resolution_data={"b":rdat},
                fibermap=fmap
                )
        
    def test_coadd(self):
        """Test coaddition"""
        nspec, nwave = 3, 10
        s1 = self._random_spectra(nspec, nwave)
        self.assertEqual(s1.flux['b'].shape[0], nspec)

        #- All the same targets, coadded in place
        s1.fibermap['TARGETID'] == 10
        coadd(s1)
        self.assertEqual(s1.flux['b'].shape[0], 1)
        
#   def test_coadd_coadd(self):
#       """Test re-coaddition of a coadd"""
#       nspec, nwave = 6, 10
#       s1 = self._random_spectra(nspec, nwave)
#       s1.fibermap['TARGETID'][0:nspec//2] = 11
#       s1.fibermap['TARGETID'][nspec//2:] = 22
#       self.assertEqual(s1.flux['b'].shape[0], nspec)
#
#       coadd(s1)
#       self.assertEqual(s1.flux['b'].shape[0], 2)
#
#       #- re-coadding should be ok
#       coadd(s1)
#       self.assertEqual(s1.flux['b'].shape[0], 2)

    def test_coadd_scores(self):
        """Test coaddition"""
        nspec, nwave = 10, 20
        s1 = self._random_spectra(nspec, nwave)
        s1.fibermap['TARGETID'] = np.arange(nspec) // 3
        coadd(s1)
        ntargets = len(np.unique(s1.fibermap['TARGETID']))
        self.assertEqual(s1.flux['b'].shape[0], ntargets)

        scores, comments = compute_coadd_scores(s1)

        self.assertEqual(len(scores['TARGETID']), ntargets)
        self.assertIn('MEDIAN_COADD_FLUX_B', scores.keys())

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
                'TILEID', 'FIBER', 'COADD_FIBERSTATUS', 'FLUX_R',
                'MEAN_FIBER_X', 'MEAN_FIBER_Y']:
            self.assertIn(col, cofm.colnames)

        #- but these columns should not be in the coadd
        for col in ['NIGHT', 'EXPID', 'FIBERSTATUS', 'FIBER_X', 'FIBER_Y']:
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
        for col in ['TARGETID', 'DESI_TARGET', 'COADD_FIBERSTATUS', 'FLUX_R']:
            self.assertIn(col, cofm.colnames)

        #- but these columns should not be in the coadd
        for col in ['NIGHT', 'EXPID', 'TILEID', 'FIBER', 'FIBERSTATUS',
                'FIBER_X', 'FIBER_Y', 'MEAN_FIBER_X', 'MEAN_FIBER_Y']:
            self.assertNotIn(col, cofm.colnames)

        #- the exposure-level fibermap has columns specific to individual
        #- exposures, but drops some of the target-level columns
        for col in ['TARGETID', 'TILEID', 'NIGHT', 'EXPID', 'FIBER',
                'FIBERSTATUS', 'FIBER_X', 'FIBER_Y']:
            self.assertIn(col, expfm.colnames)

        for col in ['DESI_TARGET', 'FLUX_R']:
            self.assertNotIn(col, expfm.colnames)


    def test_fiberstatus(self):
        """Test that FIBERSTATUS != 0 isn't included in coadd"""
        def _makespec(nspec, nwave):
            s1 = self._random_spectra(nspec, nwave)
            s1.flux['b'][:,:] = 1.0
            s1.ivar['b'][:,:] = 1.0
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
        self.assertEqual(s1.fibermap['COADD_FIBERSTATUS'][0], 0)
        self.assertTrue(np.all(s1.flux['b'] == 1.0))
        self.assertTrue(np.allclose(s1.ivar['b'], 1.0*nspec))

        #- Two spectra masked
        nspec, nwave = 5,10
        s1 = _makespec(nspec, nwave)
        self.assertEqual(len(s1.fibermap), nspec)

        s1.fibermap['FIBERSTATUS'][0] = fibermask.BROKENFIBER
        s1.fibermap['FIBERSTATUS'][1] = fibermask.BADFIBER
 
        coadd(s1)
        self.assertEqual(len(s1.fibermap), 1)
        self.assertEqual(s1.fibermap['COADD_NUMEXP'][0], nspec-2)
        self.assertEqual(s1.fibermap['COADD_FIBERSTATUS'][0], 0)
        self.assertTrue(np.all(s1.flux['b'] == 1.0))
        self.assertTrue(np.allclose(s1.ivar['b'], 1.0*(nspec-2)))

        #- All spectra masked
        nspec, nwave = 5,10
        s1 = _makespec(nspec, nwave)
        self.assertEqual(len(s1.fibermap), nspec)

        s1.fibermap['FIBERSTATUS'] = fibermask.BROKENFIBER
        
        coadd(s1)
        self.assertEqual(len(s1.fibermap), 1)
        self.assertEqual(s1.fibermap['COADD_NUMEXP'][0], 0)
        self.assertEqual(s1.fibermap['COADD_FIBERSTATUS'][0], fibermask.BROKENFIBER)
        self.assertTrue(np.all(s1.flux['b'] == 0.0))
        self.assertTrue(np.all(s1.ivar['b'] == 0.0))


    def test_coadd_cameras(self):
        """Test coaddition across cameras in a single spectrum"""
        # Coadd the dummy 3-camera spectrum.
        for b, nw in zip(self.spectra.bands, [2751, 2326, 2881]):
            self.assertEqual(len(self.spectra.wave[b]), nw)

        # Check flux
        coadds = coadd_cameras(self.spectra)
        self.assertEqual(len(coadds.wave['brz']), 7781)
        self.assertTrue(np.all(coadds.flux['brz'][0] == 0.5))

        # Check ivar inside and outside camera wavelength overlap regions
        tol = 0.0001
        wave = coadds.wave['brz']
        idx_overlap = (5760 <= wave) & (wave <= 5800+tol) | (7520 <= wave) & (wave <= 7620+tol)
        self.assertTrue(np.all(coadds.ivar['brz'][0][idx_overlap]  == 4.))
        self.assertTrue(np.all(coadds.ivar['brz'][0][~idx_overlap] == 2.))

        # Test exception due to misaligned wavelength grids.
        self.spectra.wave['r'] += 0.001
        with self.assertRaises(ValueError):
            coadds = coadd_cameras(self.spectra)

        self.spectra.wave['r'] -= 0.001
        coadds = coadd_cameras(self.spectra)
        self.assertEqual(len(coadds.wave['brz']), 7781)


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m desispec.test.test_coadd
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)


if __name__ == '__main__':
    unittest.main()           
