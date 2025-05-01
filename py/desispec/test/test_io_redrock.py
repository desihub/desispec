# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.io.redrock.
"""

import os
import unittest
import tempfile
import shutil

import numpy as np
import fitsio
from astropy.table import Table

from desispec.io import findfile
from desispec.io.redrock import read_redrock, read_redrock_targetcat

standard_nersc_environment = ('NERSC_HOST' in os.environ and
                              os.getenv('DESI_SPECTRO_REDUX') == '/global/cfs/cdirs/desi/spectro/redux')

class TestIORedrock(unittest.TestCase):
    """Test desispec.io.fibermap.
    """

    @classmethod
    def setUpClass(self):
        #- Create fake redrock files to read
        self.orig_env = dict()
        for key in ('DESI_SPECTRO_REDUX', 'SPECPROD'):
            self.orig_env[key] = os.getenv(key)

        self.new_env = dict()
        self.reduxdir = tempfile.mkdtemp()
        os.environ['DESI_SPECTRO_REDUX'] = self.new_env['DESI_SPECTRO_REDUX'] = self.reduxdir
        os.environ['SPECPROD'] = self.new_env['SPECPROD'] = 'blat'
        self.tileid = 1000
        self.night = 20201010

        self.ntargets = ntargets = 10
        zcat = Table()  #- redshift catalog
        zcat['TARGETID'] = np.arange(ntargets)
        zcat['Z'] = np.arange(ntargets)
        zcat['ZWARN'] = np.zeros(ntargets)
        zcat.meta['EXTNAME'] = 'REDSHIFTS'

        fm = Table()    #- fibermap
        fm['TARGETID'] = zcat['TARGETID']
        fm['TILEID'] = self.tileid
        fm['FIBER'] = np.arange(ntargets)
        fm['FIBER'][ntargets//2:]  = 500 + np.arange(ntargets//2)
        fm['TARGET_RA'] = np.random.uniform(0,10, size=ntargets)
        fm['TARGET_DEC'] = np.random.uniform(0,10, size=ntargets)
        fm.meta['EXTNAME'] = 'FIBERMAP'

        #- create files in a different specprod than $SPECPROD
        filename = findfile('redrock', tile=self.tileid, night=self.night, spectrograph=0, specprod='iotest')
        os.makedirs(os.path.dirname(filename))
        zcat[0:ntargets//2].write(filename)
        fitsio.write(filename, fm[0:ntargets//2].as_array(), extname='FIBERMAP')

        filename = findfile('redrock', tile=self.tileid, night=self.night, spectrograph=1, specprod='iotest')
        zcat[ntargets//2:].write(filename)
        fitsio.write(filename, fm[ntargets//2:].as_array(), extname='FIBERMAP')

        self.zcat = zcat
        self.fm_tiles = fm

        #- Also write to two healpix
        self.survey = 'main'
        self.program = 'dark'
        self.hpix1 = 1001
        self.hpix2 = 1002

        fm = Table(fm, copy=False)
        fm.remove_column('FIBER')
        fm['HEALPIX'] = self.hpix1
        fm['HEALPIX'][ntargets//2:] = self.hpix2
        fm['SURVEY'] = self.survey
        fm['PROGRAM'] = self.program
        self.fm_healpix = fm

        filename = findfile('redrock', healpix=self.hpix1, survey=self.survey, faprogram=self.program, specprod='iotest')
        os.makedirs(os.path.dirname(filename))
        zcat[0:ntargets//2].write(filename)
        fitsio.write(filename, fm[0:ntargets//2].as_array(), extname='FIBERMAP')

        filename = findfile('redrock', healpix=self.hpix2, survey=self.survey, faprogram=self.program, specprod='iotest')
        os.makedirs(os.path.dirname(filename))
        zcat[ntargets//2:].write(filename)
        fitsio.write(filename, fm[ntargets//2:].as_array(), extname='FIBERMAP')

    @classmethod
    def tearDownClass(self):
        for key, value in self.orig_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

        shutil.rmtree(self.reduxdir)

    def setUp(self):
        for key, value in self.new_env.items():
            os.environ[key] = value

    def tearDown(self):
        pass

    @unittest.skipUnless(standard_nersc_environment, "not at NERSC")
    def test_read_redrock(self):
        os.environ['DESI_SPECTRO_REDUX'] = self.orig_env['DESI_SPECTRO_REDUX']
        rrfile = findfile('redrock', tile=1000, night=20210517, groupname='cumulative', spectrograph=0, specprod='iron')

        zcat = read_redrock(rrfile)
        self.assertEqual(len(zcat), 500)
        self.assertIn('TARGETID', zcat.colnames)
        self.assertIn('Z', zcat.colnames)
        self.assertIn('TEMNAM00', zcat.meta)
        self.assertIn('TEMVER00', zcat.meta)

        zcat = read_redrock(rrfile, fmcols=('TARGET_RA', 'TARGET_DEC'))
        self.assertEqual(len(zcat), 500)
        self.assertIn('TARGETID', zcat.colnames)
        self.assertIn('Z', zcat.colnames)
        self.assertIn('TARGET_RA', zcat.colnames)
        self.assertIn('TARGET_DEC', zcat.colnames)
        self.assertIn('TEMNAM00', zcat.meta)
        self.assertIn('TEMVER00', zcat.meta)

    def test_read_redrock_targetcat_tiles(self):
        """Test reading redrock files based upon a tiles target catalog"""
        #- TARGETID, TILEID, FIBER
        ii = np.arange(0, self.ntargets, 2)
        tcat = self.fm_tiles['TARGETID', 'TILEID', 'FIBER'][ii]
        zcat = read_redrock_targetcat(tcat, specprod='iotest')
        self.assertTrue(np.all(zcat['TARGETID'] == tcat['TARGETID']))
        self.assertEqual(zcat.colnames, self.zcat.colnames)

        #- Only TILEID, FIBER; different order
        ii = [2,6,1,8]
        tcat = self.fm_tiles[['TILEID', 'FIBER',]][ii]
        zcat = read_redrock_targetcat(tcat, specprod='iotest')
        self.assertTrue(np.all(zcat['TARGETID'] == self.fm_tiles['TARGETID'][ii]))
        self.assertEqual(zcat.colnames, self.zcat.colnames)

        #- Only TILEID, PETAL_LOC, TARGETID; reverse order
        ii = np.arange(self.ntargets)[-1::-1]
        tcat = self.fm_tiles[['TILEID', 'TARGETID',]][ii]
        tcat['PETAL_LOC'] = self.fm_tiles['FIBER'][ii] // 500
        zcat = read_redrock_targetcat(tcat, specprod='iotest')
        self.assertTrue(np.all(zcat['TARGETID'] == tcat['TARGETID']))
        self.assertEqual(zcat.colnames, self.zcat.colnames)

        #- Test adding extra columns from fibermap
        ii = np.arange(0, self.ntargets, 2)
        tcat = self.fm_tiles['TARGETID', 'TILEID', 'FIBER'][ii]
        fmcols = ('FIBER', 'TARGET_RA', 'TARGET_DEC')
        zcat = read_redrock_targetcat(tcat, fmcols=fmcols, specprod='iotest')
        self.assertTrue(np.all(zcat['TARGETID'] == tcat['TARGETID']))
        for col in fmcols:
            self.assertTrue(np.all(zcat[col] == self.fm_tiles[col][ii]))

    def test_read_redrock_targetcat_healpix(self):
        """Test reading redrock files based upon a healpix target catalog"""
        ii = np.arange(0, self.ntargets, 2)
        tcat = self.fm_healpix['TARGETID', 'HEALPIX', 'SURVEY', 'PROGRAM'][ii]
        zcat = read_redrock_targetcat(tcat, specprod='iotest')
        self.assertTrue(np.all(zcat['TARGETID'] == tcat['TARGETID']))
        self.assertEqual(zcat.colnames, self.zcat.colnames)

        #- TODO: add setup+test for same TARGETID different PROGRAMs





