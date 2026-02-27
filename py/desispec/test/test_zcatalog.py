"""
Test desispec.zcatalog
"""

import os
import unittest
from unittest.mock import patch, call, MagicMock

import numpy as np
from astropy.table import Table, Column

from desispec.zcatalog import (find_primary_spectra, _get_survey_program_from_filename,
                               update_table_columns)


class TestZCatalog(unittest.TestCase):

    def test_find_primary_spectra(self):
        #- TARGETID ZWARN TSNR2_LRG TEST
        rows = [
           (10, 0, 100.0, 0),
           (10, 0, 200.0, 1),  # larger TSNR2_LRG = better
           (20, 4,   0.0, 1),  # only entry for this target
           (30, 4, 100.0, 0),
           (30, 0,  10.0, 1),  # zwarn=0 trumps larger TSNR2
           (40, 4, 100.0, 1),  # zwarn value doesn't matter except 0 or non-0
           (40, 8,  10.0, 0),
           (50, 8, 100.0, 1),  # zwarn value doesn't matter except 0 or non-0
           (50, 4,  10.0, 0),
           (60, 0,  10.0, 1),  # TSNR2=0 doesn't break things
           (60, 0,   0.0, 0),
           (-1, 0,  10.0, 1),  # negative TARGETIDs are ok
           (-1, 0,   0.0, 0),
        ]

        zcat = Table(rows=rows, names=('TARGETID','ZWARN','TSNR2_LRG','TEST'))
        n, best = find_primary_spectra(zcat)
        self.assertTrue( np.all(zcat['TEST'] == best) )
        self.assertTrue(isinstance(n, np.ndarray))
        self.assertTrue(isinstance(best, np.ndarray))

        # also works for numpy array input
        n, best = find_primary_spectra(np.array(zcat))
        self.assertTrue( np.all(zcat['TEST'] == best) )

        # custom column name
        zcat.rename_column('TSNR2_LRG', 'BLAT')
        n, best = find_primary_spectra(zcat, sort_column='BLAT')
        self.assertTrue( np.all(zcat['TEST'] == best) )

        # custom column name, even if TSNR2_LRG is present don't use it
        zcat['TSNR2_LRG'] = np.zeros(len(zcat))
        n, best = find_primary_spectra(zcat, sort_column='BLAT')
        self.assertTrue( np.all(zcat['TEST'] == best) )

    def test__get_survey_program_from_filename(self):
        survey, program = _get_survey_program_from_filename('/desi/spectro/redux/specprod/zcatalog/v1/zall-main-dark.fits')
        self.assertEqual(survey, 'main')
        self.assertEqual(program, 'dark')
        survey, program = _get_survey_program_from_filename('ztile-sv3-bright-cumulative.fits')
        self.assertEqual(survey, 'sv3')
        self.assertEqual(program, 'bright')

    @patch('desispec.zcatalog.log')
    def test_update_table_columns_default(self, mock_log):
        """Test update_table_columns with columns_list = None.
        """
        rows = 5
        targetid = Column(np.arange(rows, dtype=np.int64), name='TARGETID')
        survey = Column(np.array(['main']*rows), name='SURVEY')
        program = Column(np.array(['dark']*rows), name='PROGRAM')
        desi_target = Column(np.array([0]*rows), name='DESI_TARGET')
        bgs_target = Column(np.array([0]*rows), name='BGS_TARGET')
        numobs_init = Column(np.array([0]*rows), name='NUMOBS_INIT')
        plate_ra = Column(np.array([0]*rows), name='PLATE_RA')
        plate_dec = Column(np.array([0]*rows), name='PLATE_DEC')
        tsnr2_lrg = Column(np.array([0]*rows), name='TSNR2_LRG')
        zcat_nspec = Column(np.array([0]*rows), name='ZCAT_NSPEC')
        zcat_primary = Column(np.array([0]*rows), name='ZCAT_PRIMARY')
        t = Table([targetid, survey, program,
                   numobs_init, plate_ra, plate_dec, desi_target, bgs_target,
                   tsnr2_lrg, zcat_nspec, zcat_primary])
        self.assertListEqual(t.colnames,
                             ['TARGETID', 'SURVEY', 'PROGRAM', 'NUMOBS_INIT',
                              'PLATE_RA', 'PLATE_DEC', 'DESI_TARGET', 'BGS_TARGET',
                              'TSNR2_LRG', 'ZCAT_NSPEC', 'ZCAT_PRIMARY'])
        t2 = update_table_columns(t)
        self.assertListEqual(t2.colnames,
                             ['TARGETID', 'SURVEY', 'PROGRAM', 'NUMOBS_INIT',
                              'PLATE_RA', 'PLATE_DEC', 'TSNR2_LRG', 'ZCAT_NSPEC',
                              'ZCAT_PRIMARY', 'DESI_TARGET', 'BGS_TARGET'])
        mock_log.debug.assert_has_calls([call("columns_list is None"),])

    @patch('desispec.zcatalog.log')
    def test_update_table_columns_user(self, mock_log):
        """Test update_table_columns with columns_list = user-supplied list.
        """
        rows = 5
        targetid = Column(np.arange(rows, dtype=np.int64), name='TARGETID')
        survey = Column(np.array(['main']*rows), name='SURVEY')
        program = Column(np.array(['dark']*rows), name='PROGRAM')
        desi_target = Column(np.array([0]*rows), name='DESI_TARGET')
        bgs_target = Column(np.array([0]*rows), name='BGS_TARGET')
        numobs_init = Column(np.array([0]*rows), name='NUMOBS_INIT')
        plate_ra = Column(np.array([0]*rows), name='PLATE_RA')
        plate_dec = Column(np.array([0]*rows), name='PLATE_DEC')
        tsnr2_lrg = Column(np.array([0]*rows), name='TSNR2_LRG')
        zcat_nspec = Column(np.array([0]*rows), name='ZCAT_NSPEC')
        zcat_primary = Column(np.array([0]*rows), name='ZCAT_PRIMARY')
        t = Table([targetid, survey, program,
                   numobs_init, plate_ra, plate_dec, desi_target, bgs_target,
                   tsnr2_lrg, zcat_nspec, zcat_primary])
        self.assertListEqual(t.colnames,
                             ['TARGETID', 'SURVEY', 'PROGRAM', 'NUMOBS_INIT',
                              'PLATE_RA', 'PLATE_DEC', 'DESI_TARGET', 'BGS_TARGET',
                              'TSNR2_LRG', 'ZCAT_NSPEC', 'ZCAT_PRIMARY'])

        # subset but in standard order
        columns = ['TARGETID', 'SURVEY', 'PROGRAM', 'ZCAT_PRIMARY']
        t2 = update_table_columns(t, columns_list=columns)
        self.assertListEqual(t2.colnames, columns)

        # non-standard order
        columns = ['DESI_TARGET', 'TARGETID', 'PLATE_DEC', 'PLATE_RA']
        t2 = update_table_columns(t, columns_list=columns)
        self.assertListEqual(t2.colnames, columns)

        t2 = update_table_columns(t, columns_list=['TARGETID', 'SURVEY',
                                                   'PROGRAM', 'ZCAT_PRIMARY'])
        self.assertListEqual(t2.colnames,
                             ['TARGETID', 'SURVEY', 'PROGRAM', 'ZCAT_PRIMARY'])
        with self.assertRaises(KeyError):
            t2 = update_table_columns(t, columns_list=['TARGETID', 'SURVEY',
                                                       'PROGRAM', 'FOOBAR'])
        mock_log.debug.assert_has_calls([call("columns_list is user-supplied"),])
