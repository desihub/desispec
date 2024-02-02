# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.io.fibermap.
"""
import os
import unittest

import numpy as np
from astropy.io import fits
from astropy.table import Table
import fitsio

from ..io.fibermap import empty_fibermap, read_fibermap, write_fibermap, find_fiberassign_file

class TestIO(unittest.TestCase):
    """Test desispec.io.fibermap.
    """

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

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
