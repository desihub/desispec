# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec assumptions about astropy behavior that change across versions.

These tests do not target a single desispec module.  They pin down the
astropy behaviors that desispec I/O silently depends upon, so that an astropy
upgrade fails loudly in CI instead of subtly in production.  See the astropy
8.0 changes to ``Table.read(strip_spaces=...)`` and to the ``chararray``
return type of ``astropy.io.fits`` string columns.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np
import fitsio
import astropy
from astropy.io import fits
from astropy.table import Table

#- strip_spaces was added to Table.read in astropy 7.1 and turned on by
#- default in astropy 8.0
_astropy_major = int(astropy.__version__.split('.')[0])
_astropy_minor = int(astropy.__version__.split('.')[1])
_has_strip_spaces = (_astropy_major, _astropy_minor) >= (7, 1)
_strips_spaces_by_default = _astropy_major >= 8


class TestAstropyCompat(unittest.TestCase):
    """Test astropy behaviors that desispec I/O relies upon.
    """

    @classmethod
    def setUpClass(cls):
        cls.testDir = tempfile.mkdtemp()
        cls.testfile = os.path.join(cls.testDir, 'test_astropy_compat.fits')

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.testDir):
            shutil.rmtree(cls.testDir)

    def tearDown(self):
        if os.path.exists(self.testfile):
            os.remove(self.testfile)

    def _write_stringtable(self, writer):
        """Write a test table with string columns using `writer` ('fitsio' or 'astropy').

        Returns the numpy array that was written.
        """
        data = np.zeros(3, dtype=[('SPECTYPE', 'S10'), ('SUBTYPE', 'S20'),
                                 ('TARGETID', 'i8')])
        data['SPECTYPE'] = [b'GALAXY', b'QSO', b'STAR']
        data['SUBTYPE'] = [b'', b'', b'M']
        data['TARGETID'] = [10, 20, 30]
        if writer == 'fitsio':
            fitsio.write(self.testfile, data, extname='ZCAT', clobber=True)
        elif writer == 'astropy':
            Table(data).write(self.testfile, format='fits', overwrite=True)
        else:
            raise ValueError(f'unknown writer {writer}')

        return data

    def test_string_columns_are_stripped(self):
        """astropy and fitsio must agree on FITS string column values.

        desispec reads FITS tables via both astropy.table.Table.read and
        desispec.io.table.read_table (fitsio).  Values from the two paths are
        compared and mixed throughout the code, so they must match.  astropy
        8.0 changed the Table.read strip_spaces default to True, which would
        break this agreement for space-padded files.
        """
        from ..io.table import read_table

        for writer in ('fitsio', 'astropy'):
            with self.subTest(writer=writer):
                self._write_stringtable(writer)
                astropy_table = Table.read(self.testfile, hdu=1)
                desispec_table = read_table(self.testfile, ext=1)
                for colname in ('SPECTYPE', 'TARGETID'):
                    a = np.asarray(astropy_table[colname]).astype(str)
                    b = np.asarray(desispec_table[colname]).astype(str)
                    self.assertTrue(np.all(a == b),
                                    f'{writer} {colname} mismatch {a} != {b}')

                #- and no trailing whitespace snuck in
                spectype = np.asarray(astropy_table['SPECTYPE']).astype(str)
                self.assertTrue(np.all(spectype == np.char.strip(spectype)))

    def test_read_table_avoids_masked_strings(self):
        """desispec.io.table.read_table must not return masked string columns.

        Table.read converts blank strings to masked values; read_table exists
        specifically to avoid that.  This test documents both behaviors so a
        change in either one is caught.
        """
        from ..io.table import read_table

        self._write_stringtable('fitsio')

        desispec_table = read_table(self.testfile, ext=1)
        subtype = desispec_table['SUBTYPE']
        self.assertFalse(hasattr(subtype, 'mask'),
                         'read_table returned a masked SUBTYPE column')
        self.assertEqual(np.asarray(subtype).astype(str)[0], '')

        #- astropy.table.Table.read still masks the blank entries; if this
        #- ever changes, desispec.io.table.read_table is no longer needed
        astropy_table = Table.read(self.testfile, hdu=1)
        self.assertTrue(hasattr(astropy_table['SUBTYPE'], 'mask'))

    @unittest.skipUnless(_has_strip_spaces,
                         'astropy < 7.1 has no Table.read strip_spaces option')
    def test_space_padded_strings(self):
        """Space-padded FITS string columns must read back stripped.

        DESI files NUL-pad their string columns, so the astropy 8.0
        strip_spaces default change is a no-op for them.  Externally produced
        files (idlutils era, some cfitsio writers) do space-pad, and desispec
        assumes those read back stripped.
        """
        self._write_stringtable('fitsio')

        #- SPECTYPE is S10, so 'GALAXY' is written with 4 bytes of padding;
        #- replace the NUL padding written by fitsio with space padding
        with open(self.testfile, 'rb') as fx:
            raw = fx.read()

        nul_padded = b'GALAXY' + b'\x00' * 4
        self.assertIn(nul_padded, raw,
                      'test assumption broken: SPECTYPE is not NUL padded')
        space_padded = b'GALAXY' + b' ' * 4
        with open(self.testfile, 'wb') as fx:
            fx.write(raw.replace(nul_padded, space_padded))

        def spectype0(**kwargs):
            table = Table.read(self.testfile, hdu=1, **kwargs)
            return np.asarray(table['SPECTYPE']).astype(str)[0]

        self.assertEqual(spectype0(strip_spaces=True), 'GALAXY')
        self.assertEqual(spectype0(strip_spaces=False), 'GALAXY    ')

        #- astropy 8.0 flipped the default from False to True
        if _strips_spaces_by_default:
            self.assertEqual(spectype0(), 'GALAXY')
        else:
            self.assertEqual(spectype0(), 'GALAXY    ')

    def test_fits_string_column_type(self):
        """Track when astropy.io.fits stops returning chararray.

        astropy 8.0 deprecated the chararray methods and warns that string
        columns will become plain ndarrays in a future version.  chararray
        strips trailing whitespace on element access; a plain ndarray does not.
        Any desispec code reading string columns through astropy.io.fits (as
        opposed to Table.read) silently depends on that stripping.  When this
        test starts failing, audit those reads.
        """
        self._write_stringtable('fitsio')

        data = fits.getdata(self.testfile, 1)
        column = data['SPECTYPE']
        self.assertEqual(type(column).__name__, 'chararray',
                         'astropy.io.fits no longer returns chararray for '
                         'string columns; audit desispec fits.getdata() reads '
                         'of string columns for trailing whitespace')

    def test_write_bintable_tunit_indices(self):
        """TUNITn must line up with TTYPEn when only some columns have units.

        desispec.io.util.write_bintable walks TTYPE1..TTYPEn and inserts each
        TUNITn after the matching TFORMn.  With units for only a subset of
        columns the TUNIT numbering is sparse, which the simpler two-column
        case in test_io.test_write_bintable does not exercise.
        """
        from ..io.util import write_bintable

        data = Table()
        data['TARGETID'] = [1, 2, 3]
        data['FLUX_R'] = [1.0, 2.0, 3.0]
        data['SPECTYPE'] = ['GALAXY', 'QSO', 'STAR']
        data['TSNR2_LRG'] = [10.0, 20.0, 30.0]

        write_bintable(self.testfile, data, extname='ZCAT',
                       units={'FLUX_R': 'nanomaggies', 'TSNR2_LRG': ''},
                       comments={'SPECTYPE': 'Spectral type'})

        header = fits.getheader(self.testfile, 'ZCAT')
        self.assertEqual(header['TTYPE1'], 'TARGETID')
        self.assertEqual(header['TTYPE2'], 'FLUX_R')
        self.assertEqual(header['TTYPE3'], 'SPECTYPE')
        self.assertEqual(header['TTYPE4'], 'TSNR2_LRG')
        self.assertEqual(header.comments['TTYPE3'], 'Spectral type')

        #- only FLUX_R gets a TUNIT; the empty-string unit is skipped
        self.assertEqual(header['TUNIT2'], 'nanomaggies')
        self.assertEqual(header.comments['TUNIT2'], 'FLUX_R units')
        for i in (1, 3, 4):
            self.assertNotIn(f'TUNIT{i}', header)

    def test_sigma_clip_returns_mask(self):
        """astropy.stats.sigma_clip must return an object with a bool .mask.

        desispec.skygradpca indexes with ``sigma_clip(flux, axis=0).mask``.
        astropy 8.0 added Masked support to sigma clipping; plain ndarray
        input must still give a MaskedArray-style .mask.
        """
        import astropy.stats

        rng = np.random.RandomState(0)
        flux = rng.normal(size=(20, 50))
        flux[5, 10] = 1000.0

        clipped = astropy.stats.sigma_clip(flux, axis=0)
        mask = clipped.mask
        self.assertEqual(mask.shape, flux.shape)
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertTrue(mask[5, 10])

        mean, median, stddev = astropy.stats.sigma_clipped_stats(flux)
        self.assertTrue(np.isfinite([mean, median, stddev]).all())
