import unittest

import numpy as np
import scipy.interpolate

from astropy.table import Table
from astropy.table.column import Column
from astropy.units import Unit

from desispec.spectra import Spectra
from desispec.io import empty_fibermap
from desispec.coaddition import (coadd, fast_resample_spectra,
                                 spectroperf_resample_spectra,
                                 coadd_fibermap, coadd_cameras,
                                 _mask_cosmics)
from desispec.specscore import compute_coadd_scores
from desispec.resolution import Resolution
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
        
    def _random_spectra(self, ns=3, nw=10, seed=None, with_mask=False,
                        bands=('b',)):

        rng = np.random.default_rng(seed)
        nres = 3 # number of resolution elts
        rdat = np.ones((ns, nres, nw))
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
        fmap["FIBER_RA"] = rng.normal(loc=10, scale=0.1, size=ns)
        fmap["FIBER_DEC"] = rng.normal(loc=0, scale=0.1, size=ns)
        #- dummy scores
        scores = dict()
        scores['BLAT'] = np.zeros(ns)
        scores['FOO'] = np.ones(ns)
        scores['BAR'] = np.arange(ns)
        # create overlapping/pixel aligned wavelength intervals
        wave_b = np.linspace(4000, 6000, nw)
        wave_b_edge = wave_b[int(0.7 * nw)]
        step = wave_b[1] - wave_b[0]
        wave_r = np.linspace(wave_b_edge, wave_b_edge + (nw - 1) * step,
                             nw)
        wave_r_edge = wave_r[int(0.7 * nw)]
        wave_z = np.linspace(wave_r_edge, wave_r_edge + (nw - 1) * step,
                             nw)
        wave = {'b': wave_b,
                'r': wave_r,
                'z': wave_z}
        flux = {}
        ivar = {}
        resolution_data = {}
        if with_mask:
            mask = {}
        else:
            mask = None
        for band in bands:
            flux[band] = rng.uniform(0, 1, size=(ns,nw))
            ivar[band] = rng.uniform(0, 1, size=(ns,nw))
            if with_mask:
                mask[band] = np.zeros((ns, nw), dtype=int)
            resolution_data[band] = rdat.copy()
            
        return Spectra(
                bands=bands,
                wave=wave,
                flux=flux,
                ivar=ivar,
                mask=mask,
                resolution_data=resolution_data,
                fibermap=fmap,
                scores=scores,
                )
    def test_cosmic_masking(self):
        rng = np.random.default_rng(133)
        npix, nspec = 10000, 30
        wave = np.arange(npix)
        ivar = rng.uniform(0, 1,
                           size=(nspec, npix)) * (1 + 100 * np.arange(nspec)[:, None])
        model0 = 0.01 * (wave - (wave)**2 / npix) + rng.uniform(size=npix)
        flux = model0 + rng.normal(size=(nspec, npix)) / np.sqrt(ivar)
        COSMIC = 1e6
        flux[2, 100] = COSMIC
        flux[0, 130] = COSMIC
        mask = np.zeros((nspec, npix), dtype=int)
        ivarjj_masked = ivar * 1
        cosmics_nsig = 4
        _mask_cosmics(wave, flux, ivar, mask, np.arange(nspec),
                      ivarjj_masked, tid=1,
                      cosmics_nsig=cosmics_nsig)
        # we mask pixel and 1 neighbor around hence 3 pixel per cosmic
        self.assertEqual((ivarjj_masked == 0).sum(), 6)
        self.assertEqual((flux[ivarjj_masked == 0] == COSMIC).sum(),
                         2)
    
    def test_coadd(self):
        """Test coaddition"""
        nspec, nwave = 3, 10
        s1 = self._random_spectra(nspec, nwave)
        self.assertEqual(s1.flux['b'].shape[0], nspec)
        self.assertIsInstance(s1.scores, Table)

        #- All the same targets, coadded in place
        s1.fibermap['TARGETID'] = 10
        coadd(s1)
        self.assertEqual(s1.flux['b'].shape[0], 1)
        self.assertIsInstance(s1.scores, Table)

    def test_coadd_masked(self):
        """Test coaddition when all spectra have certain wavelength range masked
        In this case we still prefer to preserve ivar like nothing happened
        """
        nspec, nwave = 3, 10
        s1 = self._random_spectra(nspec, nwave, with_mask=True)
        maskpix = 5
        s1.mask['b'][:,:maskpix] = 1
        # All the same targets, coadded in place
        s1.fibermap['TARGETID'] = 10
        ivar0 = s1.ivar['b'].copy()
        resol0 = s1.resolution_data['b'].copy()[0]
        coadd(s1)
        self.assertTrue(np.all(np.isfinite(s1.mask['b'])))
        self.assertTrue(np.all(np.isfinite(s1.flux['b'])))
        self.assertTrue(np.allclose(s1.resolution_data['b'] , resol0))
        self.assertTrue(np.all(s1.ivar['b'] == np.sum(ivar0, axis=0)))
        self.assertTrue(np.all(s1.mask['b'][0][:maskpix] != 0))

    def test_coadd_single(self):
        """Test coaddition of a single spectrum which should be no-op"""
        nspec, nwave = 1, 10
        s1 = self._random_spectra(nspec, nwave)
        spec0 = s1.flux['b'] * 1
        ivar0 = s1.ivar['b'] * 1
        resmat = s1.resolution_data['b'] * 1
        #- All the same targets, coadded in place
        s1.fibermap['TARGETID'] = 10
        coadd(s1)
        self.assertTrue(np.allclose(s1.flux['b'], spec0))
        self.assertTrue(np.all(s1.ivar['b'] == ivar0))
        self.assertTrue(np.all(s1.resolution_data['b'] == resmat))

    def test_coadd_single_mask(self):
        """Test coaddition with a masked pixel triggering #2372"""
        nspec, nwave = 1, 10
        # check middle pixel and edge pixel
        for mpix in [0,5]: 
            s1 = self._random_spectra(nspec, nwave, with_mask=True)
            s1.mask['b'][0, mpix] = 1
            s1.ivar['b'][0, mpix] = 0
            nonmasked = s1.mask['b'][0] == 0
            resmat1 = Resolution(s1.resolution_data['b'][0] * 1)
            # All the same targets, coadded in place
            s1.fibermap['TARGETID'] = 10
            coadd(s1)
            resmat2 = Resolution(s1.resolution_data['b'][0] * 1)
            modvec = np.ones(nwave)
            # Here we are testing that after model spectra are
            # the same as before coadd (outside the masked pixel)
            mod1 = resmat1@modvec
            mod2 = resmat2@modvec
            self.assertTrue(np.allclose(mod1[nonmasked], mod2[nonmasked]))
            self.assertTrue(s1.mask['b'][0, mpix] > 0)
            self.assertTrue(s1.ivar['b'][0, mpix] == 0)
            
    def test_coadd_cameras_single_mask(self):
        """Test coaddition with a masked pixel triggering #2372
        Now with coadd_cameras
        """
        nspec, nwave = 1, 10
        s1 = self._random_spectra(nspec, nwave, with_mask=True)
        spec0 = s1.flux['b'][0] * 1
        ivar0 = s1.ivar['b'][0] * 1
        resmat1 = Resolution(s1.resolution_data['b'][0] * 1)
        for mpix in [0,5]:
            s1.mask['b'][:, mpix] = 1
            nonmask = s1.mask['b'][0] == 0
            #- All the same targets, coadded in place
            s1.fibermap['TARGETID'] = 10
            modvec = np.ones(nwave)
            s2 = coadd_cameras(s1)
            resmat2 = Resolution(s2.resolution_data['b'][0] * 1)
            mod1 = resmat1@modvec
            mod2 = resmat2@modvec
            self.assertTrue(np.allclose(s2.flux['b'][0][nonmask], spec0[nonmask]))
            self.assertTrue(np.all(s2.ivar['b'][0][nonmask] == ivar0[nonmask]))
            # temporarily disabled
            self.assertTrue(np.all((mod1 == mod2)[nonmask]))
        
    def test_coadd_full_mask(self):
        """
        Test coadd with one spectrum fully masked
        """
        nspec, nwave = 2, 30
        s1 = self._random_spectra(nspec, nwave, with_mask=True)
        flux0 = s1.flux['b'][0] * 1
        rng = np.random.default_rng(4343)
        ivar = rng.uniform(size=s1.ivar['b'].shape)
        resol = rng.uniform(size=s1.resolution_data['b'].shape)        
        s1.ivar['b'] = ivar * 1
        s1.resolution_data['b'] = resol * 1
        # fully mask second exposure
        s1.mask['b'][1, :] = 1
        # All the same targets, coadded in place
        s1.fibermap['TARGETID'] = [10] * nspec
        coadd(s1)
        # check that the output of the coadd equals the first spectrum
        self.assertTrue(np.allclose(flux0, s1.flux['b'][0]))
        self.assertTrue(np.allclose(ivar[0], s1.ivar['b'][0]))
        self.assertTrue(np.allclose(resol[0], s1.resolution_data['b'][0]))
        

    def test_coadd_resolution(self):
        """Test proper behaviour of resolution matrix
        i.e. if all input spectra were D_i = R_i * M
        coadd must satisfy the same condition
        protection against #2372 """
        nspec, nwave = 20, 30
        s1 = self._random_spectra(nspec, nwave, with_mask=True)
        rng = np.random.default_rng(4343)
        ivar = rng.uniform(size=s1.ivar['b'].shape)
        # completely mask some fraction of pixels
        ivar[rng.uniform(size=s1.ivar['b'].shape) < 0.01] = 0
        s1.ivar['b'] = ivar
        model0 = rng.uniform(size=nwave)
        # random resolution matrix
        resol = rng.uniform(size=s1.resolution_data['b'].shape)
        s1.resolution_data['b'] = resol
        for i in range(nspec):
            s1.flux['b'][i] = Resolution(resol[i])@model0
        # All the same targets, coadded in place
        s1.fibermap['TARGETID'] = [10]*nspec
        coadd(s1, cosmics_nsig=1e10)
        resmat2 = Resolution(s1.resolution_data['b'][0])
        resmod = resmat2@model0
        self.assertTrue(np.allclose(resmod, s1.flux['b'][0]))
        
    def test_coadd_cameras_resolution(self):
        """Test proper behaviour of resolution matrix
        i.e. if all input spectra were D_i = R_i * M
        coadd must satisfy the same condition
        protection against #2372 
        Here we just ignore the pixels touched by the spectrum edges from either
        arm.
        """
        nspec, nwave = 20, 100
        bands = ['b', 'r', 'z']
        rng = np.random.default_rng(4343)
        s1 = self._random_spectra(nspec, nwave, with_mask=True, bands=bands)
        s1.fibermap['TARGETID'] = [10] * nspec
        s2 = coadd_cameras(s1, cosmics_nsig=1e10)
        model0_brz = rng.uniform(1, 2, size=s2.wave['brz'].size)
        edge_nmask = 2 
        # we will ignore pixels next to the edges of spectrum
        edge_mask = np.zeros(len(model0_brz), dtype=bool)
        step = s2.wave['brz'][1] - s2.wave['brz'][0]
        for band in bands:
            edge_mask = edge_mask | (np.abs(s2.wave['brz'] -
                                            s1.wave[band][0]) <
                                     (edge_nmask + 0.1) * step)
            edge_mask = edge_mask | (np.abs(s2.wave['brz'] -
                                            s1.wave[band][-1]) <
                                     (edge_nmask + 0.1) * step)
            
        model0 = {}
        for band in bands:
            ivar = rng.uniform(size=s1.ivar[band].shape)
            # completely mask some fraction of pixels
            ivar[rng.uniform(size=s1.ivar[band].shape) < 0.01] = 0
            s1.ivar[band] = ivar
            resol = rng.uniform(size=s1.resolution_data[band].shape)
            # random resolution matrix
            s1.resolution_data[band] = resol
            model0[band] = scipy.interpolate.interp1d(s2.wave['brz'],
                                                      model0_brz, kind='nearest',
                                                      fill_value='extrapolate',
                                                      bounds_error=False)(
                                                          s1.wave[band])
            for i in range(nspec):
                s1.flux[band][i] = Resolution(resol[i]) @ model0[band]
        s2 = coadd_cameras(s1, cosmics_nsig=1e10)
        resmat2 = Resolution(s2.resolution_data['brz'][0])
        resmod = resmat2@model0_brz
        self.assertTrue(np.allclose(resmod[~edge_mask],
                                    s2.flux['brz'][0][~edge_mask]))

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
        self.assertIsInstance(s1.scores, Table)
        s1.fibermap['TARGETID'] = np.arange(nspec) // 3
        coadd(s1)
        ntargets = len(np.unique(s1.fibermap['TARGETID']))
        self.assertEqual(s1.flux['b'].shape[0], ntargets)
        self.assertIsInstance(s1.scores, Table)

        scores, comments = compute_coadd_scores(s1)

        self.assertEqual(len(scores['TARGETID']), ntargets)
        self.assertIn('MEDIAN_COADD_FLUX_B', scores.keys())

    def test_coadd_slice(self):
        """Test slices of coaddition"""
        from desispec.coaddition import coadd, coadd_cameras
        nspec, nwave = 6, 10
        s1 = self._random_spectra(nspec, nwave)
        s1.fibermap['TARGETID'] = [1,1,2,2,3,3]
        ntarget = len(set(s1.fibermap['TARGETID']))

        coadd(s1) #- in place coaddition
        self.assertEqual(len(s1.fibermap), ntarget)

        s2 = s1[0:2]
        self.assertEqual(len(s2.fibermap), 2)
        self.assertTrue(np.all(s2.fibermap['TARGETID'] == s1.fibermap['TARGETID'][0:2]))

        s3 = coadd_cameras(s1)
        self.assertTrue(np.all(s3.fibermap['TARGETID'] == s1.fibermap['TARGETID']))
        s4 = coadd_cameras(s2)
        self.assertTrue(np.all(s4.fibermap['TARGETID'] == s2.fibermap['TARGETID']))

    def test_spectroperf_resample(self):
        """Test spectroperf_resample"""
        s1 = self._random_spectra(1,20)
        wave = np.linspace(4000, 6000, 10)
        s2 = spectroperf_resample_spectra(s1,wave=wave)
        
    def test_fast_resample(self):
        """Test fast_resample"""
        s1 = self._random_spectra(1,20)
        wave = np.linspace(4000, 6000, 10)
        s2 = fast_resample_spectra(s1,wave=wave)

    def test_coadd_fibermap_onetile(self):
        """Test coadding a fibermap of a single tile"""
        #- one tile, 3 targets, 2 exposures on 2 nights
        fm = Table()
        fm['TARGETID'] = [111,111,222,222,333,333]
        fm['DESI_TARGET'] = [4,4,8,8,16,16]
        fm['TILEID'] = [1,1,1,1,1,1]
        fm['NIGHT'] = [20201220,20201221]*3
        fm['MJD'] = 55555.0 + np.arange(6)
        fm['EXPID'] = [10,20,11,21,12,22]
        fm['FIBER'] = [5,6,]*3
        fm['FIBERSTATUS'] = [0,0,0,0,0,0]
        fm['FIBER_X'] = [1.1, 2.1]*3
        fm['FIBER_Y'] = [10.2, 5.3]*3
        fm['DELTA_X'] = [1.1, 2.1]*3
        fm['DELTA_Y'] = [10.2, 5.3]*3
        fm['FIBER_RA'] = [5.5, 6.6]*3
        fm['FIBER_DEC'] = [7.7, 8.8]*3
        fm['FLUX_R'] = np.ones(6)
        fm['PSF_TO_FIBER_SPECFLUX'] = np.ones(6)
        cofm, expfm = coadd_fibermap(fm, onetile=True)

        #- Single tile coadds include these in the coadded fibermap
        for col in ['TARGETID', 'DESI_TARGET',
                'TILEID', 'FIBER', 'COADD_FIBERSTATUS', 'FLUX_R',
                'MEAN_FIBER_X', 'MEAN_FIBER_Y']:
            self.assertIn(col, cofm.colnames)

        #- but these columns should not be in the coadd
        for col in ['NIGHT', 'EXPID', 'FIBERSTATUS', 'FIBER_X', 'FIBER_Y',
                    'FIBER_RA', 'FIBER_DEC', 'DELTA_X', 'DELTA_Y',
                    'PSF_TO_FIBER_SPECFLUX']:
            self.assertNotIn(col, cofm.colnames)

        #- the exposure-level fibermap has columns specific to individual
        #- exposures, but drops some of the target-level columns
        for col in ['TARGETID', 'TILEID', 'NIGHT', 'EXPID', 'FIBER',
                'FIBERSTATUS', 'FIBER_X', 'FIBER_Y']:
            self.assertIn(col, expfm.colnames)

        for col in ['DESI_TARGET', 'FLUX_R']:
            self.assertNotIn(col, expfm.colnames)

        #- IN_COADD_B/R/Z should be only new columns in expfm
        newcols = set(expfm.colnames) - set(fm.colnames)
        self.assertEqual(newcols, set(['IN_COADD_B', 'IN_COADD_R', 'IN_COADD_Z']))

        #- The expfm should be in the same order as the input fm
        self.assertTrue(np.all(fm['NIGHT'] == expfm['NIGHT']))
        self.assertTrue(np.all(fm['EXPID'] == expfm['EXPID']))
        self.assertTrue(np.all(fm['FIBER'] == expfm['FIBER']))

        #- but expfm and fm are actually different at the column level
        fm['NIGHT'][0] = 999
        self.assertNotEqual(expfm['NIGHT'][0], 999)

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

    def test_coadd_fibermap_units(self):
        """Test that units aren't dropped during coaddition"""
        fm = empty_fibermap(10)
        fm['TARGET_RA'] = Column(fm['TARGET_RA'], unit='deg')
        fm['FIBER_RA'] = Column(fm['FIBER_RA'], unit='deg')
        self.assertEqual(fm['TARGET_RA'].unit, Unit('deg'))
        self.assertEqual(fm['FIBER_RA'].unit, Unit('deg'))

        cofm, expfm = coadd_fibermap(fm)
        self.assertEqual(cofm['TARGET_RA'].unit, Unit('deg'))
        self.assertEqual(expfm['FIBER_RA'].unit, Unit('deg'))

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

    def test_coadd_fibermap_ra_wrap(self):
        """Test coadding fibermap near RA=0 boundary"""
        #- differences for (FIBER_RA - TARGET_RA)
        delta_ra = np.array([-0.2, 0.0, 0.2])
        ref_std_ra = np.float32(np.std(delta_ra) * 3600)
        n = len(delta_ra)
        dec = 60.0

        #- one tile, 1 target
        fm = Table()
        fm['TARGETID'] = [111,] * n
        fm['DESI_TARGET'] = [4,] * n
        fm['TILEID'] = [1,] * n
        fm['NIGHT'] = [20201220,] * n
        fm['EXPID'] = 10 + np.arange(n)
        fm['FIBER'] = [5,] * n
        fm['FIBERSTATUS'] = [0,] * n
        fm['TARGET_DEC'] = [dec,]*n
        fm['FIBER_DEC'] = [dec,]*n

        for ra in (359.9, 0, 0.1, 10):
            fm['TARGET_RA'] = [ra,] * n
            fm['FIBER_RA'] = ra + delta_ra / np.cos(np.radians(dec))

            cofm, expfm = coadd_fibermap(fm, onetile=True)
            self.assertAlmostEqual(cofm['MEAN_FIBER_RA'][0], ra, msg=f'mean(RA) at {ra=} {dec=}')
            self.assertAlmostEqual(cofm['STD_FIBER_RA'][0], ref_std_ra, msg=f'std(RA) at {ra=} {dec=}')

    def test_mean_std_ra_dec(self):
        """Test calc_mean_std_ra"""
        from desispec.coaddition import calc_mean_std_ra_dec
        delta = np.array([-0.2, 0.0, 0.2])
        std_delta = np.std(delta) * 3600

        for ra in (0.0, 0.1, 1.0, 179.9, 180.0, 180.1, 359.0, 359.9):
            for dec in (-60, 0, 60):
                ras = (ra + delta/np.cos(np.radians(dec)) + 360) % 360
                decs = dec * np.ones(len(ras))
                mean_ra, std_ra, mean_dec, std_dec = calc_mean_std_ra_dec(ras, decs)
                self.assertAlmostEqual(mean_ra, ra,
                                       msg=f'mean RA at {ra=} {dec=}')
                self.assertAlmostEqual(std_ra, std_delta,
                                       msg=f'std RA at {ra=} {dec=}')
                self.assertAlmostEqual(mean_dec, dec,
                                       msg=f'mean dec at {ra=} {dec=}')
                self.assertAlmostEqual(std_dec, 0.0,
                                       msg=f'std dec at {ra=} {dec=}')

        #- Also check that std_dec doesn't depend upon RA
        mean_ra, std_ra, mean_dec, std_dec = calc_mean_std_ra_dec(delta, delta)
        self.assertAlmostEqual(std_dec, std_delta)
        mean_ra, std_ra, mean_dec, std_dec = calc_mean_std_ra_dec(180+delta, delta)
        self.assertAlmostEqual(std_dec, std_delta)

        #- Confirm that 0 <= RA < 360
        ras = [359.8, 0.1]  # should average to 359.95, not -0.05
        decs = [0, 0]
        mean_ra, std_ra, mean_dec, std_dec = calc_mean_std_ra_dec(ras, decs)
        self.assertAlmostEqual(mean_ra, 359.95)  # not -0.05


    def test_coadd_fibermap_mjd_night(self):
        """Test adding MIN/MAX/MEAN_MJD columns"""
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
        self.assertNotIn('MIN_MJD', cofm.colnames)

        #- also with MJD
        fm['MJD'] = 55555 + np.arange(nspec)
        cofm, expfm = coadd_fibermap(fm, onetile=True)
        self.assertEqual(cofm['MIN_MJD'][0], np.min(fm['MJD']))
        self.assertEqual(cofm['MAX_MJD'][0], np.max(fm['MJD']))
        self.assertEqual(cofm['MEAN_MJD'][0], np.mean(fm['MJD']))

        #- with some fibers masked:
        #- MIN/MAX/MEAN_MJD based only upon good ones
        fm['FIBERSTATUS'][0] = fibermask.BADFIBER     #- bad
        fm['FIBERSTATUS'][1] = fibermask.RESTRICTED   #- ok
        fm['FIBERSTATUS'][2] = fibermask.BADAMPR      #- ok for fibermap
        ok = np.ones(nspec, dtype=bool)
        ok[0] = False
        cofm, expfm = coadd_fibermap(fm, onetile=True)
        self.assertEqual(cofm['MIN_MJD'][0], np.min(fm['MJD'][ok]))
        self.assertEqual(cofm['MAX_MJD'][0], np.max(fm['MJD'][ok]))
        self.assertEqual(cofm['MEAN_MJD'][0], np.mean(fm['MJD'][ok]))

        #- multiple targets
        fm['TARGETID'][0:2] += 1
        fm['FIBERSTATUS'] = 0
        cofm, expfm = coadd_fibermap(fm, onetile=True)
        self.assertEqual(cofm['MIN_MJD'][0],    np.min(fm['MJD'][0:2]))
        self.assertEqual(cofm['MAX_MJD'][0],    np.max(fm['MJD'][0:2]))
        self.assertEqual(cofm['MEAN_MJD'][0],  np.mean(fm['MJD'][0:2]))
        self.assertEqual(cofm['MIN_MJD'][1],    np.min(fm['MJD'][2:]))
        self.assertEqual(cofm['MAX_MJD'][1],    np.max(fm['MJD'][2:]))
        self.assertEqual(cofm['MEAN_MJD'][1],  np.mean(fm['MJD'][2:]))

        #- one target completely masked, but MJD cols still filled
        fm['TARGETID'] = np.arange(nspec, dtype=int)//2
        fm['FIBERSTATUS'] = 0
        fm['FIBERSTATUS'][0:2] = fibermask.BADFIBER
        cofm, expfm = coadd_fibermap(fm, onetile=True)
        self.assertTrue(np.all(cofm['MEAN_MJD'] != 0))
        self.assertTrue(np.all(cofm['MIN_MJD'] != 0))
        self.assertTrue(np.all(cofm['MAX_MJD'] != 0))


    def test_coadd_targetmask(self):
        """Test coadding SV1/SV3/DESI_TARGET with varying bits"""
        nspec = 4
        fm = Table()
        fm['TARGETID'] = [111, 111, 222, 222]
        fm['TILEID'] = 100 * np.ones(nspec, dtype=int)
        fm['FIBERSTATUS'] = np.zeros(nspec, dtype=int)
        fm['DESI_TARGET'] = 4 * np.ones(nspec, dtype=int)
        fm['DESI_TARGET'][1] |= 8
        fm['DESI_TARGET'][3] |= 16

        fm['CMX_TARGET'] = fm['DESI_TARGET'].copy()
        fm['SV1_DESI_TARGET'] = fm['DESI_TARGET'].copy()
        fm['SV2_DESI_TARGET'] = fm['DESI_TARGET'].copy()
        fm['SV3_DESI_TARGET'] = fm['DESI_TARGET'].copy()
        fm['SV1_MWS_TARGET'] = fm['DESI_TARGET'].copy() + 32
        fm['SV2_MWS_TARGET'] = fm['DESI_TARGET'].copy() + 32
        fm['SV3_MWS_TARGET'] = fm['DESI_TARGET'].copy() + 32
        fm['SV1_BGS_TARGET'] = fm['DESI_TARGET'].copy() + 64
        fm['SV2_BGS_TARGET'] = fm['DESI_TARGET'].copy() + 64
        fm['SV3_BGS_TARGET'] = fm['DESI_TARGET'].copy() + 64

        cofm, expfm = coadd_fibermap(fm, onetile=True)
        # first target has bitmasks 4+8=12
        self.assertEqual(cofm['DESI_TARGET'][0], 12)
        self.assertEqual(cofm['CMX_TARGET'][0], 12)
        self.assertEqual(cofm['SV1_DESI_TARGET'][0], 12)
        self.assertEqual(cofm['SV2_DESI_TARGET'][0], 12)
        self.assertEqual(cofm['SV3_DESI_TARGET'][0], 12)
        self.assertEqual(cofm['SV1_MWS_TARGET'][0], 12+32)
        self.assertEqual(cofm['SV2_MWS_TARGET'][0], 12+32)
        self.assertEqual(cofm['SV3_MWS_TARGET'][0], 12+32)
        self.assertEqual(cofm['SV1_BGS_TARGET'][0], 12+64)
        self.assertEqual(cofm['SV2_BGS_TARGET'][0], 12+64)
        self.assertEqual(cofm['SV3_BGS_TARGET'][0], 12+64)

        # second target has bitmasks 4+16=20
        self.assertEqual(cofm['DESI_TARGET'][1], 20)
        self.assertEqual(cofm['CMX_TARGET'][1], 20)
        self.assertEqual(cofm['SV1_DESI_TARGET'][1], 20)
        self.assertEqual(cofm['SV2_DESI_TARGET'][1], 20)
        self.assertEqual(cofm['SV3_DESI_TARGET'][1], 20)
        self.assertEqual(cofm['SV1_MWS_TARGET'][1], 20+32)
        self.assertEqual(cofm['SV2_MWS_TARGET'][1], 20+32)
        self.assertEqual(cofm['SV3_MWS_TARGET'][1], 20+32)
        self.assertEqual(cofm['SV1_BGS_TARGET'][1], 20+64)
        self.assertEqual(cofm['SV2_BGS_TARGET'][1], 20+64)
        self.assertEqual(cofm['SV3_BGS_TARGET'][1], 20+64)

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

        #- All spectra masked but for different reasons
        nspec, nwave = 3,10
        s1 = _makespec(nspec, nwave)
        s1.fibermap['FIBERSTATUS'][0] = fibermask.BROKENFIBER
        s1.fibermap['FIBERSTATUS'][1] = fibermask.BADPOSITION
        s1.fibermap['FIBERSTATUS'][2] = fibermask.BADFLAT
        coadd(s1)
        self.assertEqual(s1.fibermap['COADD_NUMEXP'][0], 0)
        self.assertEqual(s1.fibermap['COADD_FIBERSTATUS'][0],
                         fibermask.mask('BROKENFIBER|BADPOSITION|BADFLAT'))
        self.assertTrue(np.all(s1.flux['b'] == 0.0))

    def test_coadd_fiberstatus(self):
        """Tests specifically focused on COADD_FIBERSTATUS; some overlap with other tests"""
        def _make_mini_fibermap(nspec):
            fm = Table()
            fm['TARGETID'] = np.full(nspec, 111)
            fm['FIBERSTATUS'] = np.zeros(nspec, dtype=np.int32)
            return fm

        #- all spectra with FIBERSTATUS==0
        nspec = 2
        fm = _make_mini_fibermap(nspec)
        cofm, expfm = coadd_fibermap(fm)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], 0)
        self.assertEqual(cofm['COADD_NUMEXP'][0], nspec)
        self.assertIn('IN_COADD_B', expfm.colnames)
        self.assertNotIn('IN_COADD_B', cofm.colnames)
        self.assertEqual(expfm['IN_COADD_B'].dtype, bool)
        self.assertEqual(expfm['IN_COADD_R'].dtype, bool)
        self.assertEqual(expfm['IN_COADD_Z'].dtype, bool)
        self.assertEqual(list(expfm['IN_COADD_B']), [True, True])
        self.assertEqual(list(expfm['IN_COADD_R']), [True, True])
        self.assertEqual(list(expfm['IN_COADD_Z']), [True, True])

        #- One spectrum with FIBERSTATUS=BROKENFIBER
        fm = _make_mini_fibermap(nspec)
        fm['FIBERSTATUS'][0] = fibermask.BROKENFIBER
        cofm, expfm = coadd_fibermap(fm)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], 0)
        self.assertEqual(cofm['COADD_NUMEXP'][0], nspec-1)
        self.assertEqual(list(expfm['IN_COADD_B']), [False, True])
        self.assertEqual(list(expfm['IN_COADD_R']), [False, True])
        self.assertEqual(list(expfm['IN_COADD_Z']), [False, True])

        #- Both spectra with FIBERSTATUS=BROKENFIBER
        fm = _make_mini_fibermap(nspec)
        fm['FIBERSTATUS'][:] = fibermask.BROKENFIBER
        cofm, expfm = coadd_fibermap(fm)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], fibermask.BROKENFIBER)
        self.assertEqual(cofm['COADD_NUMEXP'][0], 0)
        self.assertEqual(list(expfm['IN_COADD_B']), [False, False])
        self.assertEqual(list(expfm['IN_COADD_R']), [False, False])
        self.assertEqual(list(expfm['IN_COADD_Z']), [False, False])

        #- One spectrum with FIBERSTATUS=BADAMPB
        fm = _make_mini_fibermap(nspec)
        fm['FIBERSTATUS'][0] = fibermask.BADAMPB
        cofm, expfm = coadd_fibermap(fm)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], 0)
        self.assertEqual(cofm['COADD_NUMEXP'][0], nspec)
        self.assertEqual(list(expfm['IN_COADD_B']), [False, True])
        self.assertEqual(list(expfm['IN_COADD_R']), [True, True])
        self.assertEqual(list(expfm['IN_COADD_Z']), [True, True])

        #- Both spectra with FIBERSTATUS=BADAMPB
        fm = _make_mini_fibermap(nspec)
        fm['FIBERSTATUS'][:] = fibermask.BADAMPB
        cofm, expfm = coadd_fibermap(fm)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], fibermask.BADAMPB)
        self.assertEqual(cofm['COADD_NUMEXP'][0], nspec)  #- note: nspec even though B is bad
        self.assertEqual(list(expfm['IN_COADD_B']), [False, False])
        self.assertEqual(list(expfm['IN_COADD_R']), [True, True])
        self.assertEqual(list(expfm['IN_COADD_Z']), [True, True])

        #- Both spectra bad for different reasons
        fm = _make_mini_fibermap(nspec)
        fm['FIBERSTATUS'][0] = fibermask.BROKENFIBER
        fm['FIBERSTATUS'][1] = fibermask.BADFLAT
        cofm, expfm = coadd_fibermap(fm)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], fibermask.mask('BROKENFIBER|BADFLAT'))
        self.assertEqual(cofm['COADD_NUMEXP'][0], 0)
        self.assertEqual(list(expfm['IN_COADD_B']), [False, False])
        self.assertEqual(list(expfm['IN_COADD_R']), [False, False])
        self.assertEqual(list(expfm['IN_COADD_Z']), [False, False])

        #- Spectra with different bad cameras
        fm = _make_mini_fibermap(nspec)
        fm['FIBERSTATUS'][0] = fibermask.BADAMPB
        fm['FIBERSTATUS'][1] = fibermask.BADAMPR
        cofm, expfm = coadd_fibermap(fm)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], 0)
        self.assertEqual(cofm['COADD_NUMEXP'][0], 2)  #- Z got nspec=2
        self.assertEqual(list(expfm['IN_COADD_B']), [False, True])
        self.assertEqual(list(expfm['IN_COADD_R']), [True, False])
        self.assertEqual(list(expfm['IN_COADD_Z']), [True, True])

        #- Spectra with different bad cameras for all cameras
        fm = _make_mini_fibermap(nspec)
        fm['FIBERSTATUS'][0] = fibermask.mask('BADAMPB|BADAMPZ')
        fm['FIBERSTATUS'][1] = fibermask.mask('BADAMPR')
        cofm, expfm = coadd_fibermap(fm)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], 0)
        self.assertEqual(cofm['COADD_NUMEXP'][0], 2)
        self.assertEqual(list(expfm['IN_COADD_B']), [False, True])
        self.assertEqual(list(expfm['IN_COADD_R']), [True, False])
        self.assertEqual(list(expfm['IN_COADD_Z']), [False, True])

        #- One spectrum with fiber-problem, another with camera-problem
        fm = _make_mini_fibermap(nspec)
        fm['FIBERSTATUS'][0] = fibermask.BROKENFIBER
        fm['FIBERSTATUS'][1] = fibermask.BADAMPB
        cofm, expfm = coadd_fibermap(fm)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], fibermask.BADAMPB)
        self.assertEqual(cofm['COADD_NUMEXP'][0], 1)
        self.assertEqual(list(expfm['IN_COADD_B']), [False, False])
        self.assertEqual(list(expfm['IN_COADD_R']), [False, True])
        self.assertEqual(list(expfm['IN_COADD_Z']), [False, True])

        #- Same spectrum with fiber and camera problem
        fm = _make_mini_fibermap(nspec)
        fm['FIBERSTATUS'][0] = fibermask.mask('BROKENFIBER|BADAMPB')
        cofm, expfm = coadd_fibermap(fm)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], 0)
        self.assertEqual(cofm['COADD_NUMEXP'][0], 1)
        self.assertEqual(list(expfm['IN_COADD_B']), [False, True])
        self.assertEqual(list(expfm['IN_COADD_R']), [False, True])
        self.assertEqual(list(expfm['IN_COADD_Z']), [False, True])

        #- One spec with fiber and cam problem; another with diff fiber problem
        fm = _make_mini_fibermap(nspec)
        fm['FIBERSTATUS'][0] = fibermask.mask('BROKENFIBER|BADAMPB')
        fm['FIBERSTATUS'][1] = fibermask.mask('BADFLAT')
        cofm, expfm = coadd_fibermap(fm)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], fibermask.mask('BROKENFIBER|BADFLAT|BADAMPB'))
        self.assertEqual(cofm['COADD_NUMEXP'][0], 0)
        self.assertEqual(list(expfm['IN_COADD_B']), [False, False])
        self.assertEqual(list(expfm['IN_COADD_R']), [False, False])
        self.assertEqual(list(expfm['IN_COADD_Z']), [False, False])

        #- 3 spectra cases
        nspec = 3
        fm = _make_mini_fibermap(nspec)
        cofm, expfm = coadd_fibermap(fm)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], 0)
        self.assertEqual(cofm['COADD_NUMEXP'][0], 3)

        #- 2 fiber-level problems, one camera-level problem
        fm = _make_mini_fibermap(nspec)
        fm['FIBERSTATUS'][0] = fibermask.mask('BROKENFIBER')
        fm['FIBERSTATUS'][1] = fibermask.mask('BADFLAT|BADAMPB')
        fm['FIBERSTATUS'][2] = fibermask.mask('BADAMPB')
        cofm, expfm = coadd_fibermap(fm)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], fibermask.BADAMPB)
        self.assertEqual(cofm['COADD_NUMEXP'][0], 1)
        self.assertEqual(list(expfm['IN_COADD_B']), [False, False, False])
        self.assertEqual(list(expfm['IN_COADD_R']), [False, False, True])
        self.assertEqual(list(expfm['IN_COADD_Z']), [False, False, True])

        #- 3 different camera-level problems
        fm = _make_mini_fibermap(nspec)
        fm['FIBERSTATUS'][0] = fibermask.mask('BADAMPB')
        fm['FIBERSTATUS'][1] = fibermask.mask('BADAMPR')
        fm['FIBERSTATUS'][2] = fibermask.mask('BADAMPZ')
        cofm, expfm = coadd_fibermap(fm)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], 0)
        self.assertEqual(cofm['COADD_NUMEXP'][0], 3)
        self.assertEqual(list(expfm['IN_COADD_B']), [False, True, True])
        self.assertEqual(list(expfm['IN_COADD_R']), [True, False, True])
        self.assertEqual(list(expfm['IN_COADD_Z']), [True, True, False])

        #- each camera has different count of problems (z all good)
        fm = _make_mini_fibermap(nspec)
        fm['FIBERSTATUS'][0] = fibermask.mask('BADAMPB|BADAMPR')
        fm['FIBERSTATUS'][1] = fibermask.mask('BADAMPR')
        fm['FIBERSTATUS'][2] = fibermask.mask('BADAMPR')
        cofm, expfm = coadd_fibermap(fm)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], fibermask.BADAMPR)
        self.assertEqual(cofm['COADD_NUMEXP'][0], 3)
        self.assertEqual(list(expfm['IN_COADD_B']), [False, True, True])
        self.assertEqual(list(expfm['IN_COADD_R']), [False, False, False])
        self.assertEqual(list(expfm['IN_COADD_Z']), [True, True, True])

        #- fiber problem on one spec; bad camera problem on others
        fm = _make_mini_fibermap(nspec)
        fm['FIBERSTATUS'][0] = fibermask.mask('BADFLAT')
        fm['FIBERSTATUS'][1] = fibermask.mask('BADAMPR')
        fm['FIBERSTATUS'][2] = fibermask.mask('BADAMPR')
        cofm, expfm = coadd_fibermap(fm)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], fibermask.BADAMPR)
        self.assertEqual(cofm['COADD_NUMEXP'][0], 2)
        self.assertEqual(list(expfm['IN_COADD_B']), [False, True, True])
        self.assertEqual(list(expfm['IN_COADD_R']), [False, False, False])
        self.assertEqual(list(expfm['IN_COADD_Z']), [False, True, True])

        #- all cameras flagged bad on one exposure
        fm = _make_mini_fibermap(nspec)
        fm['FIBERSTATUS'][0] = fibermask.mask('BADAMPB|BADAMPR|BADAMPZ')
        fm['FIBERSTATUS'][1] = 0
        fm['FIBERSTATUS'][2] = 0
        cofm, expfm = coadd_fibermap(fm)
        self.assertEqual(cofm['COADD_FIBERSTATUS'][0], 0)
        self.assertEqual(cofm['COADD_NUMEXP'][0], 2)  #- not 3
        self.assertEqual(list(expfm['IN_COADD_B']), [False, True, True])
        self.assertEqual(list(expfm['IN_COADD_R']), [False, True, True])
        self.assertEqual(list(expfm['IN_COADD_Z']), [False, True, True])


    def test_coadd_cameras(self):
        """Test coaddition across cameras in a single spectrum"""
        # Coadd the dummy 3-camera spectrum.
        for b, nw in zip(self.spectra.bands, [2751, 2326, 2881]):
            self.assertEqual(len(self.spectra.wave[b]), nw)

        # Check flux
        coadds = coadd_cameras(self.spectra)
        self.assertEqual(len(coadds.wave['brz']), 7781)
        self.assertTrue(np.all(coadds.flux['brz'][0] == 1))

        # Check ivar inside and outside camera wavelength overlap regions
        tol = 0.0001
        wave = coadds.wave['brz']
        idx_overlap = (5760 <= wave) & (wave <= 5800+tol) | (7520 <= wave) & (wave <= 7620+tol)
        self.assertTrue(np.allclose(coadds.ivar['brz'][0][idx_overlap], 2))
        self.assertTrue(np.all(coadds.ivar['brz'][0][~idx_overlap] == 1.))

        # Test exception due to misaligned wavelength grids.
        self.spectra.wave['r'] += 0.001
        with self.assertRaises(ValueError):
            coadds = coadd_cameras(self.spectra)

        self.spectra.wave['r'] -= 0.001
        coadds = coadd_cameras(self.spectra)
        self.assertEqual(len(coadds.wave['brz']), 7781)

        # Test coadding without resolution or mask data
        spec = self.spectra[:]  #- copy before modifying
        spec.resolution_data = None
        spec.R = None
        spec.mask = None
        coadd = coadd_cameras(spec)
