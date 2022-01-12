# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.specstatus
"""

import unittest, os
import numpy as np
from astropy.table import Table
from desispec.specstatus import update_specstatus

class TestSpecStatus(unittest.TestCase):
    """Test desispec.specstatus
    """

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def _create_tiles(self, n):
        """
        Create a test tiles table with n rows
        """
        tiles = Table()
        tiles['TILEID'] = np.arange(n, dtype=int)+1
        tiles['LASTNIGHT'] = np.ones(n, dtype=int) * 20201010
        tiles['EFFTIME_SPEC'] = np.ones(n) * 1000
        tiles['GOALTIME'] = np.ones(n) * 1000
        tiles['MINTFRAC'] = np.ones(n) * 0.85
        return tiles

    def _create_specstatus(self, n):
        """
        Create a test specstatus table with n rows
        """
        specstatus = self._create_tiles(n)
        specstatus['USER'] = 'test'
        specstatus['QA'] = 'none'
        specstatus['OVERRIDE'] = np.zeros(n, dtype=int)
        specstatus['ZDONE'] = 'false'
        specstatus['QANIGHT'] = np.zeros(n, dtype=int)
        specstatus['ARCHIVEDATE'] = np.zeros(n, dtype=int)
        return specstatus

    def test_add(self):
        """Test adding a new tile"""
        specstatus = self._create_specstatus(3)
        tiles = self._create_tiles(4)
        self.assertNotIn(4, specstatus['TILEID'])

        newstatus = update_specstatus(specstatus, tiles)
        self.assertEqual(len(newstatus), 4)
        self.assertIn(4, newstatus['TILEID'])

    def test_update(self):
        """Test updating a tile due to new LASTNIGHT"""
        specstatus = self._create_specstatus(3)
        tiles = self._create_tiles(3)
        tiles['LASTNIGHT'][0] += 1
        tiles['EFFTIME_SPEC'][0] += 1
        tiles['EFFTIME_SPEC'][1] += 2   #- but not updating LASTNIGHT for this

        orig_lastnight = specstatus['LASTNIGHT'][0]
        orig_efftime = specstatus['EFFTIME_SPEC'][0]

        newstatus = update_specstatus(specstatus, tiles)
        #- new status has updated EFFTIME_SPEC because LASTNIGHT was new
        self.assertEqual(newstatus['LASTNIGHT'][0], tiles['LASTNIGHT'][0])
        self.assertEqual(newstatus['EFFTIME_SPEC'][0], tiles['EFFTIME_SPEC'][0])

        #- but other entries are unchanged
        self.assertEqual(newstatus['LASTNIGHT'][1], specstatus['LASTNIGHT'][1])
        self.assertEqual(newstatus['EFFTIME_SPEC'][1], specstatus['EFFTIME_SPEC'][1])

        #- and original specstatus is unchanged
        self.assertEqual(specstatus['LASTNIGHT'][0], orig_lastnight)
        self.assertEqual(specstatus['EFFTIME_SPEC'][1], orig_efftime)

    def test_noqa_update(self):
        """Even if tiles has QA info, don't update specstatus with it"""
        specstatus = self._create_specstatus(3)
        tiles = self._create_tiles(3)
        tiles['QA'] = 'good'
        tiles['LASTNIGHT'] += 1
        specstatus['QA'] = 'none'

        newstatus = update_specstatus(specstatus, tiles)
        self.assertTrue(np.all(newstatus['QA'] == 'none'))

    def test_update_all(self):
        """test updating non-QA for all tiles even if LASTNIGHT isn't new"""
        specstatus = self._create_specstatus(3)
        tiles = self._create_tiles(3)
        self.assertTrue(np.all(tiles['EFFTIME_SPEC'] == specstatus['EFFTIME_SPEC']))
        specstatus['QA'] = 'none'

        tiles['EFFTIME_SPEC'] += 1  #- should be updated
        tiles['QA'] = 'good'        #- should be skipped
        newstatus = update_specstatus(specstatus, tiles, update_all=True)

        #- LASTNIGHT didn't change
        self.assertTrue(np.all(newstatus['LASTNIGHT'] == specstatus['LASTNIGHT']))
        self.assertTrue(np.all(newstatus['LASTNIGHT'] == tiles['LASTNIGHT']))

        #- but EFFTIME_SPEC did
        self.assertTrue(np.all(newstatus['EFFTIME_SPEC'] != specstatus['EFFTIME_SPEC']))
        self.assertTrue(np.all(newstatus['EFFTIME_SPEC'] == tiles['EFFTIME_SPEC']))

        #- and QA did not
        self.assertTrue(np.all(newstatus['QA'] == specstatus['QA']))
        self.assertTrue(np.all(newstatus['QA'] != tiles['QA']))

    def test_update_subset(self):
        """Test that it's ok to update a subset of the specstatus tiles"""
        specstatus = self._create_specstatus(5)
        tiles = self._create_tiles(2)
        tiles['LASTNIGHT'] += 1
        tiles['EFFTIME_SPEC'] += 1

        newstatus = update_specstatus(specstatus, tiles)
        self.assertEqual(len(newstatus), len(specstatus))
        self.assertTrue(np.all(tiles['EFFTIME_SPEC'] == newstatus['EFFTIME_SPEC'][0:2]))

        tiles['TILEID'][0] = 1000
        newstatus = update_specstatus(specstatus, tiles)
        self.assertEqual(len(newstatus), len(specstatus)+1)
        self.assertIn(1000, newstatus['TILEID'])


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)

#- run all unit tests in this file
if __name__ == '__main__':
    unittest.main()

