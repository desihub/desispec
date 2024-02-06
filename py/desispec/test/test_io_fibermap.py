# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.io.fibermap.
"""
import os
import tempfile
import unittest
from shutil import rmtree

import numpy as np
from astropy.io import fits
from astropy.table import Table
import fitsio
from desiutil.annotate import check_comment_length

from ..io.fibermap import (fibermap_columns, fibermap_comments, _set_fibermap_columns,
                           empty_fibermap, write_fibermap, read_fibermap, find_fiberassign_file)


class TestIOFibermap(unittest.TestCase):
    """Test desispec.io.fibermap.
    """

    @classmethod
    def setUpClass(cls):
        """Create unique test filename in a subdirectory.
        """
        cls.testDir = tempfile.mkdtemp()
        cls.readonlyDir = tempfile.mkdtemp()
        cls.testfile = os.path.join(cls.testDir, 'desispec_test_io.fits')
        cls.testyfile = os.path.join(cls.testDir, 'desispec_test_io.yaml')
        cls.testlog = os.path.join(cls.testDir, 'desispec_test_io.log')
        # cls.testbrfile appears to be unused by this class.
        cls.testbrfile = os.path.join(cls.testDir, 'desispec_test_io-br.fits')
        cls.origEnv = {'SPECPROD': None,
                       "DESI_ROOT": None,
                       "DESI_ROOT_READONLY": None,
                       "DESI_SPECTRO_DATA": None,
                       "DESI_SPECTRO_REDUX": None,
                       "DESI_SPECTRO_CALIB": None,
                       }
        cls.testEnv = {'SPECPROD':'dailytest',
                       "DESI_ROOT": cls.testDir,
                       "DESI_ROOT_READONLY": cls.readonlyDir,
                       "DESI_SPECTRO_DATA": os.path.join(cls.testDir, 'spectro', 'data'),
                       "DESI_SPECTRO_REDUX": os.path.join(cls.testDir, 'spectro', 'redux'),
                       "DESI_SPECTRO_CALIB": os.path.join(cls.testDir, 'spectro', 'calib'),
                       }
        cls.datadir = cls.testEnv['DESI_SPECTRO_DATA']
        cls.reduxdir = os.path.join(cls.testEnv['DESI_SPECTRO_REDUX'],
                                    cls.testEnv['SPECPROD'])
        for e in cls.origEnv:
            if e in os.environ:
                cls.origEnv[e] = os.environ[e]
            os.environ[e] = cls.testEnv[e]

    @classmethod
    def tearDownClass(cls):
        """Cleanup test files if they exist.
        """
        for testfile in [cls.testfile, cls.testyfile, cls.testbrfile, cls.testlog]:
            if os.path.exists(testfile):
                os.remove(testfile)
        for e in cls.origEnv:
            if cls.origEnv[e] is None:
                del os.environ[e]
            else:
                os.environ[e] = cls.origEnv[e]

        if os.path.isdir(cls.testDir):
            rmtree(cls.testDir)

        # reset the readonly cache
        from ..io import meta
        meta._desi_root_readonly = None

    def setUp(self):
        if os.path.isdir(self.datadir):
            rmtree(self.datadir)
        if os.path.isdir(self.reduxdir):
            rmtree(self.reduxdir)

    def tearDown(self):
        for testfile in [self.testfile, self.testyfile, self.testbrfile, self.testlog]:
            if os.path.exists(testfile):
                os.remove(testfile)

        # restore environment variables if test changed them
        for key, value in self.testEnv.items():
            os.environ[key] = value

    def test_empty_fibermap(self):
        """Test creating empty fibermap objects.
        """
        fm1 = empty_fibermap(20)
        self.assertTrue(np.all(fm1['FIBER'] == np.arange(20)))
        self.assertTrue(np.all(fm1['PETAL_LOC'] == 0))

        fm2 = empty_fibermap(25, specmin=10)
        self.assertTrue(np.all(fm2['FIBER'] == np.arange(25)+10))
        self.assertTrue(np.all(fm2['PETAL_LOC'] == 0))
        self.assertTrue(np.all(fm2['LOCATION'][0:10] == fm1['LOCATION'][10:20]))

        fm3 = empty_fibermap(10, specmin=495)
        self.assertTrue(np.all(fm3['FIBER'] == np.arange(10)+495))
        self.assertTrue(np.all(fm3['PETAL_LOC'] == [0,0,0,0,0,1,1,1,1,1]))

    def test_fibermap_rw(self):
        """Test reading and writing fibermap files.
        """
        fibermap = empty_fibermap(10)
        for key in fibermap.dtype.names:
            column = fibermap[key]
            fibermap[key] = np.random.random(column.shape).astype(column.dtype)

        write_fibermap(self.testfile, fibermap)

        fm = read_fibermap(self.testfile)
        self.assertTrue(isinstance(fm, Table))

        self.assertEqual(set(fibermap.dtype.names), set(fm.dtype.names))
        for key in fibermap.dtype.names:
            c1 = fibermap[key]
            c2 = fm[key]
            #- Endianness may change, but kind, size, shape, and values are same
            self.assertEqual(c1.shape, c2.shape)
            self.assertTrue(np.all(c1 == c2))
            if c1.dtype.kind == 'U':
                self.assertTrue(c2.dtype.kind in ('S', 'U'))
            else:
                self.assertEqual(c1.dtype.kind, c2.dtype.kind)
                self.assertEqual(c1.dtype.itemsize, c2.dtype.itemsize)

        #- read_fibermap also works with open file pointer
        with fitsio.FITS(self.testfile) as fp:
            fm1 = read_fibermap(fp)
            self.assertTrue(np.all(fm1 == fm))

        with fits.open(self.testfile) as fp:
            fm2 = read_fibermap(fp)
            self.assertTrue(np.all(fm2 == fm))

    def test_find_fibermap(self):
        '''Test finding (non)gzipped fiberassign files.
        '''
        night = 20101020
        nightdir = os.path.join(self.datadir, str(night))
        os.makedirs(nightdir)
        os.makedirs(f'{nightdir}/00012345')
        os.makedirs(f'{nightdir}/00012346')
        os.makedirs(f'{nightdir}/00012347')
        os.makedirs(f'{nightdir}/00012348')
        fafile = f'{nightdir}/00012346/fiberassign-001111.fits'
        fafilegz = f'{nightdir}/00012347/fiberassign-001122.fits'

        fx = open(fafile, 'w'); fx.close()
        fx = open(fafilegz, 'w'); fx.close()

        a = find_fiberassign_file(night, 12346)
        self.assertEqual(a, fafile)

        a = find_fiberassign_file(night, 12347)
        self.assertEqual(a, fafilegz)

        a = find_fiberassign_file(night, 12348)
        self.assertEqual(a, fafilegz)

        a = find_fiberassign_file(night, 12348, tileid=1111)
        self.assertEqual(a, fafile)

        with self.assertRaises(IOError) as ex:
            find_fiberassign_file(night, 12345)

        with self.assertRaises(IOError) as ex:
            find_fiberassign_file(night, 12348, tileid=4444)

    def test_fibermap_comment_lengths(self):
        """Automate testing of fibermap comment lengths.
        """
        f = _set_fibermap_columns()
        n_long = check_comment_length(fibermap_comments['main'], error=False)
        self.assertEqual(n_long, 0)
        n_long = check_comment_length(fibermap_comments['cmx'], error=False)
        self.assertEqual(n_long, 0)
        n_long = check_comment_length(fibermap_comments['sv3'], error=False)
        self.assertEqual(n_long, 0)

