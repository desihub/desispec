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

from desiutil.io import encode_table
from desispec.io import empty_fibermap
from desispec.io.util import add_columns

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
        self.nwave = 100
        self.nspec = 5
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
            for b in self.bands:
                self.wave[b] = np.arange(self.nwave, dtype=float)
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

        sp2 = sp1[[True,False,True,False,True]]
        for band in self.bands:
            self.assertEqual(sp2.flux[band].shape[0], 3)


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
