# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.io.exposure_tile_qa.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np
from astropy.table import Table


class TestIOExposureTileQA(unittest.TestCase):
    """Test desispec.io.exposure_tile_qa read/write round trips.
    """

    @classmethod
    def setUpClass(cls):
        cls.testDir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.testDir):
            shutil.rmtree(cls.testDir)

    def setUp(self):
        self.fiber_qa = Table()
        self.fiber_qa['TARGETID'] = np.arange(5, dtype=np.int64)
        self.fiber_qa['FIBER'] = np.arange(5, dtype=np.int32)
        self.fiber_qa['QAFIBERSTATUS'] = np.zeros(5, dtype=np.int32)
        self.fiber_qa['TSNR2_LRG'] = np.arange(5, dtype=np.float32)

        self.petal_qa = Table()
        self.petal_qa['PETAL_LOC'] = np.arange(3, dtype=np.int32)
        self.petal_qa['WORSTREADNOISE'] = np.arange(3, dtype=np.float32)

    def _roundtrip(self, writer, reader, filename, petal=True):
        """Write and read back with `writer`/`reader`, comparing contents."""
        petal_qa = self.petal_qa if petal else None
        writer(filename, self.fiber_qa, petal_qa)
        self.assertTrue(os.path.exists(filename))

        fiber_qa, read_petal_qa = reader(filename)

        self.assertEqual(fiber_qa.colnames, self.fiber_qa.colnames)
        for colname in self.fiber_qa.colnames:
            self.assertTrue(np.all(fiber_qa[colname] == self.fiber_qa[colname]),
                            f'{colname} mismatch')
        self.assertEqual(fiber_qa.meta['EXTNAME'], 'FIBERQA')

        if petal:
            self.assertEqual(read_petal_qa.colnames, self.petal_qa.colnames)
            for colname in self.petal_qa.colnames:
                self.assertTrue(np.all(read_petal_qa[colname] == self.petal_qa[colname]),
                                f'{colname} mismatch')
        else:
            self.assertIsNone(read_petal_qa)

    def test_exposure_qa_roundtrip(self):
        """Test write_exposure_qa and read_exposure_qa."""
        from ..io.exposure_tile_qa import write_exposure_qa, read_exposure_qa

        filename = os.path.join(self.testDir, 'exposure-qa-00000012.fits')
        self._roundtrip(write_exposure_qa, read_exposure_qa, filename)

    def test_exposure_qa_no_petal(self):
        """Test write_exposure_qa without a petal table."""
        from ..io.exposure_tile_qa import write_exposure_qa, read_exposure_qa

        filename = os.path.join(self.testDir, 'exposure-qa-nopetal.fits')
        self._roundtrip(write_exposure_qa, read_exposure_qa, filename,
                        petal=False)

    def test_tile_qa_roundtrip(self):
        """Test write_tile_qa and read_tile_qa."""
        from ..io.exposure_tile_qa import write_tile_qa, read_tile_qa

        filename = os.path.join(self.testDir, 'tile-qa-1234-thru20201220.fits')
        self._roundtrip(write_tile_qa, read_tile_qa, filename)

    def test_write_creates_directory(self):
        """write_exposure_qa creates the output directory if needed."""
        from ..io.exposure_tile_qa import write_exposure_qa, read_exposure_qa

        subdir = os.path.join(self.testDir, 'newdir', 'deeper')
        filename = os.path.join(subdir, 'exposure-qa-00000013.fits')
        self.assertFalse(os.path.isdir(subdir))
        write_exposure_qa(filename, self.fiber_qa)
        self.assertTrue(os.path.exists(filename))

        fiber_qa, petal_qa = read_exposure_qa(filename)
        self.assertIsNone(petal_qa)
        self.assertEqual(len(fiber_qa), len(self.fiber_qa))
