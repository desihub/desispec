# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.scripts.
"""
from __future__ import absolute_import, division
# The line above will help with 2to3 support.

import os
import unittest
from uuid import uuid4
from astropy.table import Table


class TestScripts(unittest.TestCase):
    """Test desispec.scripts.
    """

    @classmethod
    def setUpClass(cls):
        # from os import environ
        # for k in ('DESI_SPECTRO_REDUX', 'SPECPROD'):
        #     if k in environ:
        #         raise AssertionError("{0}={1} was pre-defined in the environment!".format(k, environ[k]))
        cls.environ_cache = dict()

    @classmethod
    def tearDownClass(cls):
        cls.environ_cache.clear()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def dummy_environment(self, env):
        """Set dummy environment variables for testing.

        Parameters
        ----------
        env : :class:`dict`
            Mapping of variables to values
        """
        from os import environ
        for k in env:
            if k in environ:
                self.environ_cache[k] = environ[k]
            else:
                self.environ_cache[k] = None
            environ[k] = env[k]
        return

    def test_get_completed_tiles(self):
        """
        tests desispec.scripts.submit_night.get_completed_tiles(testfile)
        to ensure that the tile selection logic matches the expectations.
        The test varies ZDONE, SURVEY, EFFTIME_SPEC, and FAPRGRM.
        """
        from desispec.scripts.submit_night import get_completed_tiles
        rows = []
        tiles_truth = []
        row_names = ['ZDONE', 'SURVEY', 'EFFTIME_SPEC', 'GOALTIME',
                     'MINTFRAC', 'FAPRGRM', 'TILEID']
        ## Test zdone always gives true
        ## nominal dark
        rows.append(['true', 'main', 1000., 1000., 0.85, 'dark', 1]) # pass
        rows.append(['true', 'sv', 1000., 1000., 0.85, 'dark', 2]) # pass
        ## med efftime backup
        rows.append(['true', 'main', 500., 1000., 0.85, 'dark', 3]) # pass
        rows.append(['true', 'sv', 500., 1000., 0.85, 'dark', 4]) # pass
        ## low efftime dark
        rows.append(['true', 'main', 10., 1000., 0.85, 'dark', 5]) # pass
        rows.append(['true', 'sv', 10., 1000., 0.85, 'dark', 6]) # pass
        ## nominal bright
        rows.append(['true', 'main', 180., 180., 0.85, 'bright', 7]) # pass
        rows.append(['true', 'sv', 180., 180., 0.85, 'bright', 8]) # pass
        ## med efftime backup
        rows.append(['true', 'main', 90., 180., 0.85, 'bright', 9]) # pass
        rows.append(['true', 'sv', 90., 180., 0.85, 'bright', 10]) # pass
        ## low efftime bright
        rows.append(['true', 'main', 10., 180., 0.85, 'bright', 11]) # pass
        rows.append(['true', 'sv', 10., 180., 0.85, 'bright', 12]) # pass
        ## nominal backup
        rows.append(['true', 'main', 60., 60., 0.85, 'backup', 13]) # pass
        rows.append(['true', 'sv', 60., 60., 0.85, 'backup', 14]) # pass
        ## med efftime backup
        rows.append(['true', 'main', 30., 60., 0.85, 'backup', 15]) # pass
        rows.append(['true', 'sv', 30., 60., 0.85, 'backup', 16]) # pass
        ## low efftime backup
        rows.append(['true', 'main', 3., 60., 0.85, 'backup', 17]) # pass
        rows.append(['true', 'sv', 3., 60., 0.85, 'backup', 18]) # pass
        tiles_truth.extend(list(range(1,19)))

        ## Test other criteria when zdone false
        ## nominal dark
        rows.append(['false', 'main', 1000., 1000., 0.85, 'dark', 21])  # fail
        rows.append(['false', 'sv', 1000., 1000., 0.85, 'dark', 22])  # pass
        ## med efftime backup
        rows.append(['false', 'main', 500., 1000., 0.85, 'dark', 23])  # fail
        rows.append(['false', 'sv', 500., 1000., 0.85, 'dark', 24])  # pass
        ## low efftime dark
        rows.append(['false', 'main', 10., 1000., 0.85, 'dark', 25])  # fail
        rows.append(['false', 'sv', 10., 1000., 0.85, 'dark', 26])  # fail
        ## nominal bright
        rows.append(['false', 'main', 180., 180., 0.85, 'bright', 27])  # fail
        rows.append(['false', 'sv', 180., 180., 0.85, 'bright', 28])  # pass
        ## med efftime backup
        rows.append(['false', 'main', 90., 180., 0.85, 'bright', 29])  # fail
        rows.append(['false', 'sv', 90., 180., 0.85, 'bright', 30])  # pass
        ## low efftime bright
        rows.append(['false', 'main', 10., 180., 0.85, 'bright', 31])  # fail
        rows.append(['false', 'sv', 10., 180., 0.85, 'bright', 32])  # fail
        ## nominal backup
        rows.append(['false', 'main', 60., 60., 0.85, 'backup', 33])  # pass
        rows.append(['false', 'sv', 60., 60., 0.85, 'backup', 34])  # pass
        ## med efftime backup
        rows.append(['false', 'main', 30., 60., 0.85, 'backup', 35])  # pass
        rows.append(['false', 'sv', 30., 60., 0.85, 'backup', 36])  # pass
        ## low efftime backup
        rows.append(['false', 'main', 3., 60., 0.85, 'backup', 37])  # fail
        rows.append(['false', 'sv', 3., 60., 0.85, 'backup', 38])  # fail
        tiles_truth.extend([22, 24, 28, 30, 33, 34, 35, 36])

        test_table = Table(names=row_names, rows=rows)
        testfile = f'test-{uuid4().hex}.ecsv'
        test_table.write(testfile, overwrite=True)
        tiles_test = list(get_completed_tiles(testfile))
        if os.path.exists(testfile):
            os.remove(testfile)
        self.assertListEqual(tiles_truth, tiles_test)

    def clear_environment(self):
        """Reset environment variables after a test.
        """
        from os import environ
        for k in self.environ_cache:
            if self.environ_cache[k] is None:
                del environ[k]
            else:
                environ[k] = self.environ_cache[k]
        self.environ_cache.clear()
        return
