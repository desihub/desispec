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

        #- move away from RA,DEC = (0,0) which is treated as bad
        fmap['TARGET_RA'] += 10
        fmap['FIBER_RA'] += 10

        self.spectra = Spectra(bands=bands,
                               wave=wave,
                               flux=flux,
                               ivar=ivar,
                               mask=None,
                               resolution_data=rdat,
                               fibermap=fmap)
        
    def _random_spectra(self, ns=3, nw=10, seed=None):

        np.random.seed(seed)
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
        #- move away from 0,0 which is treated as special (invalid) case
        fmap["TARGET_RA"] = 10
        fmap["TARGET_DEC"] = 0
        fmap["FIBER_RA"] = np.random.normal(loc=10, scale=0.1, size=ns)
        fmap["FIBER_DEC"] = np.random.normal(loc=0, scale=0.1, size=ns)
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
        s1.fibermap['TARGETID'] = 10
        coadd(s1)
        self.assertEqual(s1.flux['b'].shape[0], 1)
        
    def test_coadd_nonfatal_fibermask(self):
        """Test coaddition with non-fatal fiberstatus masks"""
        nspec, nwave = 3, 10
        s1 = self._random_spectra(nspec, nwave, seed=42)
        s2 = self._random_spectra(nspec, nwave, seed=42)
        self.assertEqual(s1.flux['b'].shape[0], nspec)
        self.assertTrue(np.all(s1.flux['b'] == s2.flux['b']))

        #- All the same targets, coadded in place
        s1.fibermap['TARGETID'] = 10
        s1.fibermap['FIBERSTATUS'] = 0
        coadd(s1)

        s2.fibermap['TARGETID'] = 10
        s2.fibermap['FIBERSTATUS'][0] = fibermask.RESTRICTED
        coadd(s2)

        #- fluxes should agree
        self.assertTrue(np.allclose(s1.flux['b'], s2.flux['b']))

        #- and so should fibermaps
        self.assertTrue(np.allclose(s1.fibermap['MEAN_FIBER_RA'], s2.fibermap['MEAN_FIBER_RA']))
        self.assertTrue(np.allclose(s1.fibermap['MEAN_FIBER_DEC'], s2.fibermap['MEAN_FIBER_DEC']))

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


    def test_coadd_fibermap_badfibers(self):
        """Test coadding a fibermap of with some excluded fibers"""
        #- one tile, 3 targets, 2 exposures on 2 nights
        fm = Table()
        fm['TARGETID'] = [111,111,222,222,333,333]
        fm['DESI_TARGET'] = [4,4,8,8,16,16]
        fm['TILEID'] = [1,1,1,1,1,1]
        fm['NIGHT'] = [20201220,20201221]*3
        fm['EXPID'] = [10,20,11,21,12,22]
        fm['FIBER'] = [5,6,]*3
        fm['FIBERSTATUS'] = [0,0,0,0,0,0]
        fm['FIBER_X'] = [1.0, 2.0]*3
        fm['FIBER_Y'] = [10.0, 5.0]*3
        fm['FLUX_R'] = np.ones(6)

        #-----
        #- the first target has a masked spectrum so its coadd is different
        fm['FIBERSTATUS'][0] = fibermask.BADFIBER
        cofm, expfm = coadd_fibermap(fm, onetile=True)
        self.assertAlmostEqual(cofm['MEAN_FIBER_X'][0], 2.0)
        self.assertAlmostEqual(cofm['MEAN_FIBER_X'][1], 1.5)
        self.assertAlmostEqual(cofm['MEAN_FIBER_X'][2], 1.5)
        self.assertAlmostEqual(cofm['MEAN_FIBER_Y'][0], 5.0)
        self.assertAlmostEqual(cofm['MEAN_FIBER_Y'][1], 7.5)
        self.assertAlmostEqual(cofm['MEAN_FIBER_Y'][2], 7.5)
        #- coadd used only the good inputs, so COADD_FIBERSTATUS=0
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], 0)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][1], 0)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][1], 0)

        #-----
        #- But if it is an non-fatal bit, the coadd is the same
        fm['FIBERSTATUS'][0] = fibermask.RESTRICTED
        cofm, expfm = coadd_fibermap(fm, onetile=True)
        self.assertAlmostEqual(cofm['MEAN_FIBER_X'][0], 1.5)
        self.assertAlmostEqual(cofm['MEAN_FIBER_X'][1], 1.5)
        self.assertAlmostEqual(cofm['MEAN_FIBER_X'][2], 1.5)
        self.assertAlmostEqual(cofm['MEAN_FIBER_Y'][0], 7.5)
        self.assertAlmostEqual(cofm['MEAN_FIBER_Y'][1], 7.5)
        self.assertAlmostEqual(cofm['MEAN_FIBER_Y'][2], 7.5)
        #- coadd used only the good inputs, so COADD_FIBERSTATUS=0
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], 0)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][1], 0)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][1], 0)

        #-----
        #- also the same for a per-amp bit
        fm['FIBERSTATUS'][0] = fibermask.BADAMPB
        fm['FIBERSTATUS'][1] = fibermask.BADAMPR
        cofm, expfm = coadd_fibermap(fm, onetile=True)
        self.assertAlmostEqual(cofm['MEAN_FIBER_X'][0], 1.5)
        self.assertAlmostEqual(cofm['MEAN_FIBER_X'][1], 1.5)
        self.assertAlmostEqual(cofm['MEAN_FIBER_X'][2], 1.5)
        self.assertAlmostEqual(cofm['MEAN_FIBER_Y'][0], 7.5)
        self.assertAlmostEqual(cofm['MEAN_FIBER_Y'][1], 7.5)
        self.assertAlmostEqual(cofm['MEAN_FIBER_Y'][2], 7.5)
        #- coadd used only the good inputs, so COADD_FIBERSTATUS=0
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], 0)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][1], 0)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][1], 0)


    def test_coadd_fibermap_radec(self):
        """Test coadding fibermap RA,DEC"""
        #- one tile, 3 targets, 2 exposures on 2 nights
        fm = Table()
        fm['TARGETID'] = [111,111,111,222,222,222]
        fm['DESI_TARGET'] = [4,4,4,8,8,8]
        fm['TILEID'] = [1,1,1,1,1,1]
        fm['NIGHT'] = [20201220,20201221]*3
        fm['EXPID'] = [10,20,11,21,12,22]
        fm['FIBER'] = [5,5,5,6,6,6]
        fm['FIBERSTATUS'] = [0,0,0,0,0,0]
        fm['TARGET_RA'] = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
        fm['TARGET_DEC'] = [0.0, 0.0, 0.0, 60.0, 60.0, 60.0]
        fm['FIBER_RA'] = [10.0, 10.1, 10.2, 20.0, 20.1, 20.2]
        fm['FIBER_DEC'] = [0.0, 0.1, 0.2, 60.0, 60.1, 60.2]

        #-----
        #- the first target has a masked spectrum so its coadd is different
        fm['FIBERSTATUS'][0] = fibermask.BADFIBER
        cofm, expfm = coadd_fibermap(fm, onetile=True)
        self.assertAlmostEqual(cofm['MEAN_FIBER_RA'][0], 10.15)
        self.assertAlmostEqual(cofm['MEAN_FIBER_RA'][1], 20.10)
        self.assertAlmostEqual(cofm['MEAN_FIBER_DEC'][0], 0.15)
        self.assertAlmostEqual(cofm['MEAN_FIBER_DEC'][1], 60.1)

        #-----
        #- invalid values are excluded even if there is no FIBERSTATUS bit set
        fm['FIBERSTATUS'] = 0
        fm['FIBER_RA'][2] = 0.0
        fm['FIBER_DEC'][2] = 0.0
        cofm, expfm = coadd_fibermap(fm, onetile=True)
        self.assertAlmostEqual(cofm['MEAN_FIBER_RA'][0], 10.05)
        self.assertAlmostEqual(cofm['MEAN_FIBER_RA'][1], 20.10)
        self.assertAlmostEqual(cofm['MEAN_FIBER_DEC'][0], 0.05)
        self.assertAlmostEqual(cofm['MEAN_FIBER_DEC'][1], 60.1)

    def test_coadd_fibermap_badradec(self):
        """Test coadding a fibermap of with bad RA,DEC values"""
        nspec = 10
        fm = Table()
        fm['TARGETID'] = 111 * np.ones(nspec, dtype=int)
        fm['DESI_TARGET'] = 4 * np.ones(nspec, dtype=int)
        fm['TILEID'] = np.ones(nspec, dtype=int)
        fm['NIGHT'] = 20201220 * np.ones(nspec, dtype=int)
        fm['EXPID'] = np.arange(nspec, dtype=int)
        fm['FIBER'] = np.ones(nspec, dtype=int)
        fm['FIBERSTATUS'] = np.zeros(nspec, dtype=int)
        fm['TARGET_RA'] = np.zeros(nspec, dtype=float)
        fm['TARGET_DEC'] = np.zeros(nspec, dtype=float)
        fm['FIBER_RA'] = np.zeros(nspec, dtype=float)
        fm['FIBER_DEC'] = np.zeros(nspec, dtype=float)

        cofm, expfm = coadd_fibermap(fm, onetile=True)

        self.assertTrue(np.allclose(cofm['MEAN_FIBER_RA'], 0.0))
        self.assertTrue(np.allclose(cofm['MEAN_FIBER_DEC'], 0.0))

    def test_coadd_fibermap_mjd_night(self):
        """Test adding MIN/MAX/MEAN_MJD and FIRST/LASTNIGHT columns"""
        nspec = 5
        fm = Table()
        fm['TARGETID'] = 111 * np.ones(nspec, dtype=int)
        fm['DESI_TARGET'] = 4 * np.ones(nspec, dtype=int)
        fm['TILEID'] = np.ones(nspec, dtype=int)
        fm['NIGHT'] = 20201220 + np.arange(nspec)
        fm['EXPID'] = np.arange(nspec, dtype=int)
        fm['FIBERSTATUS'] = np.zeros(nspec, dtype=int)

        #- with NIGHT but not MJD
        cofm, expfm = coadd_fibermap(fm, onetile=True)
        self.assertEqual(len(cofm), 1)  #- single target in this test
        self.assertEqual(cofm['FIRSTNIGHT'][0], np.min(fm['NIGHT']))
        self.assertEqual(cofm['LASTNIGHT'][0], np.max(fm['NIGHT']))
        self.assertNotIn('MIN_MJD', cofm.colnames)

        #- also with MJD
        fm['MJD'] = 55555 + np.arange(nspec)
        cofm, expfm = coadd_fibermap(fm, onetile=True)
        self.assertEqual(cofm['MIN_MJD'][0], np.min(fm['MJD']))
        self.assertEqual(cofm['MAX_MJD'][0], np.max(fm['MJD']))
        self.assertEqual(cofm['MEAN_MJD'][0], np.mean(fm['MJD']))

        #- with some fibers masked
        fm['FIBERSTATUS'][0] = fibermask.BADFIBER     #- bad
        fm['FIBERSTATUS'][1] = fibermask.RESTRICTED   #- ok
        fm['FIBERSTATUS'][2] = fibermask.BADAMPR      #- ok for fibermap
        ok = np.ones(nspec, dtype=bool)
        ok[0] = False
        cofm, expfm = coadd_fibermap(fm, onetile=True)
        self.assertEqual(cofm['FIRSTNIGHT'][0], np.min(fm['NIGHT'][ok]))
        self.assertEqual(cofm['LASTNIGHT'][0], np.max(fm['NIGHT'][ok]))
        self.assertEqual(cofm['MIN_MJD'][0], np.min(fm['MJD'][ok]))
        self.assertEqual(cofm['MAX_MJD'][0], np.max(fm['MJD'][ok]))
        self.assertEqual(cofm['MEAN_MJD'][0], np.mean(fm['MJD'][ok]))

        #- multiple targets
        fm['TARGETID'][0:2] += 1
        fm['FIBERSTATUS'] = 0
        cofm, expfm = coadd_fibermap(fm, onetile=True)
        self.assertEqual(cofm['FIRSTNIGHT'][0], np.min(fm['NIGHT'][0:2]))
        self.assertEqual(cofm['LASTNIGHT'][0],  np.max(fm['NIGHT'][0:2]))
        self.assertEqual(cofm['MIN_MJD'][0],    np.min(fm['MJD'][0:2]))
        self.assertEqual(cofm['MAX_MJD'][0],    np.max(fm['MJD'][0:2]))
        self.assertEqual(cofm['MEAN_MJD'][0],  np.mean(fm['MJD'][0:2]))
        self.assertEqual(cofm['FIRSTNIGHT'][1], np.min(fm['NIGHT'][2:]))
        self.assertEqual(cofm['LASTNIGHT'][1],  np.max(fm['NIGHT'][2:]))
        self.assertEqual(cofm['MIN_MJD'][1],    np.min(fm['MJD'][2:]))
        self.assertEqual(cofm['MAX_MJD'][1],    np.max(fm['MJD'][2:]))
        self.assertEqual(cofm['MEAN_MJD'][1],  np.mean(fm['MJD'][2:]))


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

        #- non-fatal mask bits set
        nspec, nwave = 5,10
        s1 = _makespec(nspec, nwave)
        self.assertEqual(len(s1.fibermap), nspec)

        s1.fibermap['FIBERSTATUS'] = fibermask.RESTRICTED
 
        coadd(s1)
        self.assertEqual(len(s1.fibermap), 1)
        self.assertEqual(s1.fibermap['COADD_NUMEXP'][0], nspec)
        self.assertEqual(s1.fibermap['COADD_FIBERSTATUS'][0], fibermask.RESTRICTED)
        self.assertTrue(np.all(s1.flux['b'] == 1.0))
        self.assertTrue(np.allclose(s1.ivar['b'], 1.0*(nspec)))

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
