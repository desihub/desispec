"""
tests desispec.spectra.py
"""

import os
import unittest
import shutil
import time
import copy
import warnings

import numpy as np
import numpy.testing as nt

from astropy.table import Table, vstack

_specutils_imported = True
try:
    from specutils import SpectrumList, Spectrum1D
    # from astropy.units import Unit
    # from astropy.nddata import InverseVariance, StdDevUncertainty
except ImportError:
    _specutils_imported = False

from desiutil.io import encode_table
from desispec.io import empty_fibermap
from desispec.io.util import add_columns
import desispec.coaddition

# Import all functions from the module we are testing.
from desispec.spectra import *
from desispec.io.spectra import *


class TestSpectra(unittest.TestCase):

    def setUp(self):
        #- catch specific warnings so that we can find and fix
        # warnings.filterwarnings("error", ".*did not parse as fits unit.*")

        #- Test data and files to work with
        self.fileio = "test_spectra.fits"
        self.fileappend = "test_spectra_append.fits"
        self.filebuild = "test_spectra_build.fits"
        self.meta = {
            "KEY1" : "VAL1",
            "KEY2" : "VAL2"
        }
        self.nwave = 101
        self.nspec = 6
        self.ndiag = 3

        fmap = empty_fibermap(self.nspec)
        fmap = add_columns(fmap,
                           ['NIGHT', 'EXPID', 'TILEID'],
                           [np.int32(0), np.int32(0), np.int32(0)],
                           )

        for s in range(self.nspec):
            fmap[s]["TARGETID"] = 456 + s
            fmap[s]["FIBER"] = 123 + s
            fmap[s]["NIGHT"] = s
            fmap[s]["EXPID"] = s
        self.fmap1 = encode_table(fmap)
        self.efmap1 = vstack([self.fmap1, self.fmap1])

        fmap = empty_fibermap(self.nspec)
        fmap = add_columns(fmap,
                           ['NIGHT', 'EXPID', 'TILEID'],
                           [np.int32(0), np.int32(0), np.int32(0)],
                           )

        for s in range(self.nspec):
            fmap[s]["TARGETID"] = 789 + s
            fmap[s]["FIBER"] = 200 + s
            fmap[s]["NIGHT"] = 1000
            fmap[s]["EXPID"] = 1000+s
        self.fmap2 = encode_table(fmap)
        self.efmap2 = vstack([self.fmap2, self.fmap2])

        for s in range(self.nspec):
            fmap[s]["TARGETID"] = 1234 + s
            fmap[s]["FIBER"] = 300 + s
            fmap[s]["NIGHT"] = 2000
            fmap[s]["EXPID"] = 2000+s
        self.fmap3 = encode_table(fmap)
        self.efmap3 = vstack([self.fmap3, self.fmap3])

        self.bands = ["b", "r", "z"]

        self.wave = {}
        self.flux = {}
        self.ivar = {}
        self.mask = {}
        self.res = {}
        self.extra = {}

        for s in range(self.nspec):
            self.wave['b'] = np.linspace(3500, 5800, self.nwave, dtype=float)
            self.wave['r'] = np.linspace(5570, 7870, self.nwave, dtype=float)
            self.wave['z'] = np.linspace(7640, 9940, self.nwave, dtype=float)
            for b in self.bands:
                self.flux[b] = np.repeat(np.arange(self.nspec, dtype=float),
                    self.nwave).reshape( (self.nspec, self.nwave) ) + 3.0
                self.ivar[b] = 1.0 / self.flux[b]
                self.mask[b] = np.tile(np.arange(2, dtype=np.uint32),
                    (self.nwave * self.nspec) // 2).reshape( (self.nspec, self.nwave) )
                self.res[b] = np.zeros( (self.nspec, self.ndiag, self.nwave),
                    dtype=np.float64)
                self.res[b][:,1,:] = 1.0
                self.extra[b] = {}
                self.extra[b]["FOO"] = self.flux[b]

        self.scores = dict(BLAT=np.arange(self.nspec), FOO=np.ones(self.nspec))
        self.scores_comments = dict(BLAT='blat blat', FOO='foo foo')
        self.extra_catalog = Table()
        self.extra_catalog['A'] = np.arange(self.nspec)
        self.extra_catalog['B'] = np.ones(self.nspec)

    def tearDown(self):
        if os.path.exists(self.fileio):
            os.remove(self.fileio)
        if os.path.exists(self.fileappend):
            os.remove(self.fileappend)
        if os.path.exists(self.filebuild):
            os.remove(self.filebuild)
        pass

    def verify(self, spec, fmap):
        for key, val in self.meta.items():
            assert(key in spec.meta)
            assert(spec.meta[key] == val)
        nt.assert_array_equal(spec.fibermap, fmap)
        for band in self.bands:
            nt.assert_array_almost_equal(spec.wave[band], self.wave[band])
            nt.assert_array_almost_equal(spec.flux[band], self.flux[band])
            nt.assert_array_almost_equal(spec.ivar[band], self.ivar[band])
            nt.assert_array_equal(spec.mask[band], self.mask[band])
            nt.assert_array_almost_equal(spec.resolution_data[band], self.res[band])
            if spec.extra is not None:
                for key, val in self.extra[band].items():
                    nt.assert_array_almost_equal(spec.extra[band][key], val)
        if spec.extra_catalog is not None:
            assert(np.all(spec.extra_catalog == self.extra_catalog))

    def test_io(self):

        # manually create the spectra and write
        spec = Spectra(bands=self.bands, wave=self.wave, flux=self.flux,
            ivar=self.ivar, mask=self.mask, resolution_data=self.res,
            fibermap=self.fmap1, meta=self.meta, extra=self.extra)

        self.verify(spec, self.fmap1)

        path = write_spectra(self.fileio, spec)
        assert(path == os.path.abspath(self.fileio))

        # read back in and verify
        comp = read_spectra(self.fileio)
        self.verify(comp, self.fmap1)

        # test writing/reading with scores
        spec.scores = self.scores
        path = write_spectra(self.fileio, spec)
        comp = read_spectra(self.fileio)
        self.verify(comp, self.fmap1)

        # ... and reading/writing with scores + comments
        spec.scores_comments = self.scores_comments
        path = write_spectra(self.fileio, spec)
        comp = read_spectra(self.fileio)
        self.verify(comp, self.fmap1)

        # test I/O with the extra_catalog HDU enabled
        spec = Spectra(bands=self.bands, wave=self.wave, flux=self.flux,
            ivar=self.ivar, mask=self.mask, resolution_data=self.res,
            fibermap=self.fmap1, meta=self.meta, extra=self.extra,
            extra_catalog=self.extra_catalog)

        path = write_spectra(self.fileio, spec)
        assert(path == os.path.abspath(self.fileio))

        comp = read_spectra(self.fileio)
        self.assertTrue(comp.extra_catalog is not None)
        self.verify(comp, self.fmap1)

        # read_spectra finds files even with wrong gzip extension
        if self.fileio.endswith('fits'):
            sp = read_spectra(self.fileio+'.gz')    # finds it anywayk
        elif self.fileio.endswith('fits.gz'):
            sp = read_spectra(self.fileio[:-3])     # finds it anywayk
        else:
            raise ValueError(f'Unrecognized extension for {self.fileio=}')

    def test_read_targetids(self):
        """Test reading while filtering by targetid"""

        # manually create the spectra and write
        spec = Spectra(bands=self.bands, wave=self.wave, flux=self.flux,
            ivar=self.ivar, mask=self.mask, resolution_data=self.res,
            fibermap=self.fmap1, meta=self.meta, extra=self.extra)

        write_spectra(self.fileio, spec)

        # read subset in same order as file
        ii = [2,3]
        spec_subset = spec[ii]
        targetids = spec_subset.fibermap['TARGETID']
        comp_subset = read_spectra(self.fileio, targetids=targetids)
        self.assertTrue(np.all(spec_subset.fibermap['TARGETID'] == comp_subset.fibermap['TARGETID']))
        self.assertTrue(np.allclose(spec_subset.flux['b'], comp_subset.flux['b']))
        self.assertTrue(np.allclose(spec_subset.ivar['r'], comp_subset.ivar['r']))
        self.assertTrue(np.all(spec_subset.mask['z'] == comp_subset.mask['z']))
        self.assertEqual(len(comp_subset.R['b']), len(ii))
        self.assertEqual(comp_subset.R['b'][0].shape, (self.nwave, self.nwave))

        # read subset in different order than original file
        ii = [3, 1]
        spec_subset = spec[ii]
        targetids = spec_subset.fibermap['TARGETID']
        comp_subset = read_spectra(self.fileio, targetids=targetids)
        self.assertTrue(np.all(spec_subset.fibermap['TARGETID'] == comp_subset.fibermap['TARGETID']))
        self.assertTrue(np.allclose(spec_subset.flux['b'], comp_subset.flux['b']))
        self.assertTrue(np.allclose(spec_subset.ivar['r'], comp_subset.ivar['r']))
        self.assertTrue(np.all(spec_subset.mask['z'] == comp_subset.mask['z']))

        # read subset in different order than original file, with repeats and missing targetids
        spec.fibermap['TARGETID'] = (np.arange(self.nspec) // 2) * 2 # [0, 0, 2, 2, 4, 4] for nspec=6
        spec.fibermap['TARGETID'][-1] = 5
        write_spectra(self.fileio, spec)
        targetids = [2,10,4,4,4,0,0]
        comp_subset = read_spectra(self.fileio, targetids=targetids)

        # targetid 2 appears 2x because it is in the input file twice
        # targetid 4 appears 3x because it was requested 3 times
        # targetid 0 appears 4x because it was in the input file twice and requested twice
        # and targetid 0 is at the end of comp_subset, not the beginning like the file
        # targetid 5 was not requested.
        # targetid 10 doesn't appear because it wasn't in the input file, ok
        self.assertListEqual(comp_subset.fibermap['TARGETID'].tolist(),
                             [2, 2, 4, 4, 4, 0, 0, 0, 0])

        # make sure coadded spectra with FIBERMAP vs. EXP_FIBERMAP works
        tid = 555666
        spec.fibermap['TARGETID'][0:2] = tid
        desispec.coaddition.coadd(spec)  #- in place-coadd
        write_spectra(self.fileio, spec)

        comp_subset = read_spectra(self.fileio, targetids=[tid,])
        self.assertEqual(len(comp_subset.fibermap), 1)
        self.assertEqual(len(comp_subset.exp_fibermap), 2)
        self.assertTrue(np.all(comp_subset.fibermap['TARGETID'] == tid))
        self.assertTrue(np.all(comp_subset.exp_fibermap['TARGETID'] == tid))

    def test_read_rows(self):
        """Test reading specific rows"""

        # manually create the spectra and write
        spec = Spectra(bands=self.bands, wave=self.wave, flux=self.flux,
            ivar=self.ivar, mask=self.mask, resolution_data=self.res,
            fibermap=self.fmap1, meta=self.meta, extra=self.extra)

        write_spectra(self.fileio, spec)

        rows = [1,3]
        subset = read_spectra(self.fileio, rows=rows)
        self.assertTrue(np.all(spec.fibermap[rows] == subset.fibermap))

        with self.assertRaises(ValueError):
            subset = read_spectra(self.fileio, rows=rows, targetids=[1,2])

    def test_read_columns(self):
        """test reading while subselecting columns"""
        # manually create the spectra and write
        spec = Spectra(bands=self.bands, wave=self.wave, flux=self.flux,
            ivar=self.ivar, mask=self.mask, resolution_data=self.res,
            fibermap=self.fmap1, meta=self.meta)

        write_spectra(self.fileio, spec)

        test = read_spectra(self.fileio, select_columns=dict(FIBERMAP=('TARGETID', 'FIBER')))
        self.assertIn('TARGETID', test.fibermap.colnames)
        self.assertIn('FIBER', test.fibermap.colnames)
        self.assertIn('FLUX_R', spec.fibermap.colnames)
        self.assertNotIn('FLUX_R', test.fibermap.colnames)

    def test_read_skip_hdus(self):
        """test reading while skipping some HDUs"""
        # manually create the spectra and write
        spec = Spectra(bands=self.bands, wave=self.wave, flux=self.flux,
            ivar=self.ivar, mask=self.mask, resolution_data=self.res,
            fibermap=self.fmap1, meta=self.meta, exp_fibermap=self.fmap1)

        write_spectra(self.fileio, spec)

        test = read_spectra(self.fileio, skip_hdus=('MASK', 'RESOLUTION'))
        self.assertIsNone(test.mask)
        self.assertIsNone(test.R)
        self.assertIsNotNone(test.fibermap) #- fibermap not skipped

        test = read_spectra(self.fileio, skip_hdus=('EXP_FIBERMAP', 'SCORES', 'RESOLUTION'))
        self.assertIsNone(test.exp_fibermap)
        self.assertIsNone(test.scores)
        self.assertIsNone(test.R)
        self.assertIsNotNone(test.fibermap) #- fibermap not skipped

    def test_empty(self):

        spec = Spectra(meta=self.meta)

        other = {}
        for b in self.bands:
            other[b] = Spectra(bands=[b], wave={b : self.wave[b]},
                flux={b : self.flux[b]}, ivar={b : self.ivar[b]},
                mask={b : self.mask[b]}, resolution_data={b : self.res[b]},
                fibermap=self.fmap1, meta=self.meta, extra={b : self.extra[b]})

        for b in self.bands:
            spec.update(other[b])

        self.verify(spec, self.fmap1)

        dummy = Spectra()
        spec.update(dummy)

        self.verify(spec, self.fmap1)

        path = write_spectra(self.filebuild, spec)

    def test_updateselect(self):
        spec = Spectra(bands=self.bands, wave=self.wave, flux=self.flux, ivar=self.ivar,
            mask=self.mask, resolution_data=self.res, fibermap=self.fmap1,
            meta=self.meta, extra=self.extra)

        other = Spectra(bands=self.bands, wave=self.wave, flux=self.flux, ivar=self.ivar,
            mask=self.mask, resolution_data=self.res, fibermap=self.fmap2,
            meta=self.meta, extra=self.extra)

        spec.update(other)

        path = write_spectra(self.fileappend, spec)
        assert(path == os.path.abspath(self.fileappend))

        comp = read_spectra(self.fileappend)

        nights = list(range(self.nspec))

        nig = comp.select(nights=nights)
        self.verify(nig, self.fmap1)

        nig = comp.select(nights=nights, invert=True)
        self.verify(nig, self.fmap2)

        #- scores and extra_catalog in select+update
        spec = Spectra(bands=self.bands, wave=self.wave, flux=self.flux, ivar=self.ivar,
            mask=self.mask, resolution_data=self.res, fibermap=self.fmap1,
            meta=self.meta, extra=self.extra, extra_catalog=self.extra_catalog, scores=self.scores)

        nsel = 2
        expids = list(range(nsel))
        #- return_index option
        specsel, inds = spec.select(exposures=expids, return_index=True)
        nt.assert_array_equal(inds, np.arange(nsel))

        self.assertEqual(len(specsel.extra_catalog), nsel)
        nt.assert_array_equal(specsel.extra_catalog.dtype, self.extra_catalog.dtype)
        self.assertEqual(len(specsel.scores['BLAT']), nsel)
        nt.assert_array_equal(specsel.scores['BLAT'].dtype, self.scores['BLAT'].dtype)

        ntot = 4
        specsel.update(spec[nsel:ntot])
        self.assertEqual(specsel.num_spectra(), ntot)
        self.assertEqual(len(specsel.scores['BLAT']), ntot)
        self.assertEqual(len(specsel.extra_catalog), ntot)

        #- Behavior of update when fibermaps differ, and FIBER or EXPID is not there
        fibermap_coadd = copy.deepcopy(self.fmap1)
        fibermap_coadd.remove_columns(['FIBER', 'EXPID', 'NIGHT'])
        fibermap_coadd.add_column(10, name='NUM_EXPID')
        spec_coadd = Spectra(bands=self.bands, wave=self.wave, flux=self.flux, ivar=self.ivar,
            mask=self.mask, resolution_data=self.res, meta=self.meta, extra=self.extra,
            fibermap=fibermap_coadd, scores=None)
        spec.update(spec_coadd)
        self.assertEqual(spec.num_spectra(), 2*self.nspec)
        nt.assert_array_equal(spec.fibermap.dtype, self.fmap1.dtype)
        nt.assert_array_equal(spec.fibermap['NIGHT'][self.nspec:], 0)
        nt.assert_array_equal(spec.fibermap['TARGETID'][0:self.nspec], spec.fibermap['TARGETID'][self.nspec:])

    def test_stack(self):
        """Test desispec.spectra.stack"""
        sp1 = Spectra(bands=self.bands, wave=self.wave, flux=self.flux, ivar=self.ivar,
            mask=self.mask, resolution_data=self.res,
            fibermap=self.fmap1, exp_fibermap=self.efmap1,
            meta=self.meta, extra=self.extra, scores=self.scores,
            extra_catalog=self.extra_catalog)

        sp2 = Spectra(bands=self.bands, wave=self.wave, flux=self.flux, ivar=self.ivar,
            mask=self.mask, resolution_data=self.res,
            fibermap=self.fmap2, exp_fibermap=self.efmap2,
            meta=self.meta, extra=self.extra, scores=self.scores,
            extra_catalog=self.extra_catalog)

        sp3 = Spectra(bands=self.bands, wave=self.wave, flux=self.flux, ivar=self.ivar,
            mask=self.mask, resolution_data=self.res,
            fibermap=self.fmap3, exp_fibermap=self.efmap3,
            meta=self.meta, extra=self.extra, scores=self.scores,
            extra_catalog=self.extra_catalog)

        spx = stack([sp1, sp2, sp3])
        for band in self.bands:
            self.assertEqual(spx.flux[band].shape[0], 3*self.nspec)
            self.assertEqual(spx.flux[band].shape[1], self.nwave)
            self.assertEqual(spx.ivar[band].shape[0], 3*self.nspec)
            self.assertEqual(spx.mask[band].shape[0], 3*self.nspec)
            self.assertEqual(spx.resolution_data[band].shape[0], 3*self.nspec)
            self.assertTrue(np.all(spx.flux[band][0:self.nspec] == sp1.flux[band]))
            self.assertTrue(np.all(spx.ivar[band][0:self.nspec] == sp1.ivar[band]))

        self.assertEqual(len(spx.fibermap), 3*self.nspec)
        self.assertEqual(len(spx.exp_fibermap), 3*2*self.nspec)
        self.assertEqual(len(spx.extra_catalog), 3*self.nspec)

        #- Stacking also works if optional params are None
        sp1 = Spectra(bands=self.bands, wave=self.wave, flux=self.flux, ivar=self.ivar)
        sp2 = Spectra(bands=self.bands, wave=self.wave, flux=self.flux, ivar=self.ivar)
        sp3 = Spectra(bands=self.bands, wave=self.wave, flux=self.flux, ivar=self.ivar)
        spx = stack([sp1, sp2, sp3])

    def test_slice(self):
        """Test desispec.spectra.__getitem__"""
        sp1 = Spectra(bands=self.bands, wave=self.wave, flux=self.flux, ivar=self.ivar,
            mask=self.mask, resolution_data=self.res,
            fibermap=self.fmap1, exp_fibermap=self.efmap1,
            meta=self.meta, extra=self.extra, scores=self.scores,
            extra_catalog=self.extra_catalog)

        sp2 = sp1[0:self.nspec-1]
        for band in self.bands:
            self.assertEqual(sp2.flux[band].shape[0], self.nspec-1)
            self.assertEqual(sp2.ivar[band].shape[0], self.nspec-1)
            self.assertEqual(sp2.mask[band].shape[0], self.nspec-1)
            self.assertEqual(sp2.resolution_data[band].shape[0], self.nspec-1)
            self.assertEqual(len(sp2.fibermap), self.nspec-1)
            self.assertEqual(len(sp2.exp_fibermap), 2*(self.nspec-1))
            self.assertEqual(len(sp2.extra_catalog), self.nspec-1)
            self.assertEqual(sp2.extra[band]['FOO'].shape, sp2.flux[band].shape)

        self.assertEqual(len(sp2.scores['BLAT']), self.nspec-1)

        sp2 = sp1[1:self.nspec]
        for band in self.bands:
            self.assertEqual(sp2.flux[band].shape[0], self.nspec-1)
            self.assertEqual(sp2.ivar[band].shape[0], self.nspec-1)
            self.assertEqual(sp2.mask[band].shape[0], self.nspec-1)
            self.assertEqual(sp2.resolution_data[band].shape[0], self.nspec-1)
            self.assertEqual(len(sp2.fibermap), self.nspec-1)
            self.assertEqual(len(sp2.exp_fibermap), 2*(self.nspec-1))
            self.assertEqual(len(sp2.extra_catalog), self.nspec-1)

        #- slicing also works when various optional elements are None
        sp1 = Spectra(bands=self.bands, wave=self.wave, flux=self.flux, ivar=self.ivar)
        sp2 = sp1[1:self.nspec]

        #- single element index promotes to slice
        sp2 = sp1[1]

        #- Other types of numpy-style slicing with indexes and booleans
        sp2 = sp1[[1,3]]
        for band in self.bands:
            self.assertEqual(sp2.flux[band].shape[0], 2)

        sp2 = sp1[[True, False, True, False, True, False]]
        for band in self.bands:
            self.assertEqual(sp2.flux[band].shape[0], 3)

    @unittest.skipUnless(_specutils_imported, "Unable to import specutils.")
    def test_to_specutils(self):
        """Test conversion to a specutils object.
        """
        sp1 = Spectra(bands=self.bands, wave=self.wave, flux=self.flux, ivar=self.ivar,
            mask=self.mask, resolution_data=self.res,
            fibermap=self.fmap1, exp_fibermap=self.efmap1,
            meta=self.meta, extra=self.extra, scores=self.scores,
            extra_catalog=self.extra_catalog)
        sl = sp1.to_specutils()
        self.assertEqual(sl[0].meta['single'], sp1._single)
        self.assertTrue((sl[0].mask == (sp1.mask[self.bands[0]] != 0)).all())
        self.assertTrue((sl[1].flux.value == sp1.flux[sp1.bands[1]]).all())

    @unittest.skipUnless(_specutils_imported, "Unable to import specutils.")
    def test_from_specutils(self):
        """Test conversion from a specutils object.
        """
        sp1 = Spectra(bands=self.bands, wave=self.wave, flux=self.flux, ivar=self.ivar,
            mask=self.mask, resolution_data=self.res,
            fibermap=self.fmap1, exp_fibermap=self.efmap1,
            meta=self.meta, extra=self.extra, scores=self.scores,
            extra_catalog=self.extra_catalog)
        spectrum_list = sp1.to_specutils()
        sp2 = Spectra.from_specutils(spectrum_list)
        self.assertListEqual(sp1.bands, sp2.bands)
        self.assertTrue((sp1.flux[self.bands[0]] == sp2.flux[self.bands[0]]).all())
        self.assertTrue((sp1.ivar[self.bands[1]] == sp2.ivar[self.bands[1]]).all())
        self.assertTrue((sp1.mask[self.bands[2]] == sp2.mask[self.bands[2]]).all())
        self.assertDictEqual(sp1.meta, sp2.meta)

    @unittest.skipUnless(_specutils_imported, "Unable to import specutils.")
    def test_from_specutils_coadd(self):
        """Test conversion from a Spectrum1D object representing a coadd across cameras.
        """
        sp0 = Spectra(bands=self.bands, wave=self.wave, flux=self.flux, ivar=self.ivar,
            mask=self.mask, resolution_data=self.res,
            fibermap=self.fmap1, exp_fibermap=self.efmap1,
            meta=self.meta, extra=None, scores=self.scores,
            extra_catalog=self.extra_catalog)
        sp1 = desispec.coaddition.coadd_cameras(sp0)
        spectrum_list = sp1.to_specutils()
        sp2 = Spectra.from_specutils(spectrum_list[0])
        self.assertEqual(sp2.bands[0], 'brz')
        self.assertListEqual(sp1.bands, sp2.bands)
        self.assertTrue((sp1.flux[sp1.bands[0]] == sp2.flux[sp2.bands[0]]).all())
        self.assertTrue((sp1.ivar[sp1.bands[0]] == sp2.ivar[sp2.bands[0]]).all())
        self.assertTrue((sp1.mask[sp1.bands[0]] == sp2.mask[sp2.bands[0]]).all())
        self.assertDictEqual(sp1.meta, sp2.meta)


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
