"""
Test desispec.zcatalog
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, call, MagicMock

import numpy as np
from astropy.table import Table, Column

from desispec.zcatalog import (find_primary_spectra, find_target_priority,
                               _get_survey_program_from_filename,
                               update_table_columns, create_summary_catalog)
from desispec.io.util import write_bintable


class TestZCatalog(unittest.TestCase):

    def test_find_target_priority(self):
        #- TARGETID 100: secondary(main) < primary(sv1) < primary(special);
        #- primary beats secondary regardless of survey, then special beats sv1
        #- TARGETID 200: primary(sv1) beats secondary(main), survey doesn't matter
        #- TARGETID 300: single row, trivially wins
        t = Table()
        t['TARGETID']         = [100, 100, 100, 200, 200, 300]
        t['SURVEY']           = ['main', 'sv1', 'special', 'sv1', 'main', 'main']
        t['SCND_TARGET']      = [5, 0, 0, 0, 3, 0]
        t['SV1_SCND_TARGET']  = [0, 0, 0, 0, 0, 0]
        t['SV2_SCND_TARGET']  = [0, 0, 0, 0, 0, 0]
        t['SV3_SCND_TARGET']  = [0, 0, 0, 0, 0, 0]
        t['TARGET_RA']        = [10.0, 11.0, 12.0, 20.0, 21.0, 30.0]

        targets, idx = find_target_priority(t)
        self.assertTrue(np.all(targets == [100, 200, 300]))
        self.assertEqual(idx[0], 2)  # TARGETID=100 -> row 2 (primary, special)
        self.assertEqual(idx[1], 3)  # TARGETID=200 -> row 3 (primary, sv1)
        self.assertEqual(idx[2], 5)  # TARGETID=300 -> only row

        #- No SURVEY column at all (as in a single-survey coadd_fibermap/pixgroup call):
        #- primary-vs-secondary still applies; ties fall back to first occurrence
        t2 = Table()
        t2['TARGETID'] = [5, 5, 5, 6]
        t2['SCND_TARGET'] = [0, 7, 0, 0]
        t2['TARGET_RA'] = [1.0, 2.0, 3.0, 4.0]
        targets2, idx2 = find_target_priority(t2)
        self.assertTrue(np.all(targets2 == [5, 6]))
        self.assertEqual(idx2[0], 0)  # first of the two primary rows for TARGETID=5
        self.assertEqual(idx2[1], 3)

        #- All rows secondary -> survey tier still breaks the tie
        t3 = Table()
        t3['TARGETID'] = [9, 9, 9]
        t3['SURVEY'] = ['sv2', 'main', 'special']
        t3['SCND_TARGET'] = [1, 1, 1]
        targets3, idx3 = find_target_priority(t3)
        self.assertEqual(idx3[0], 1)  # main wins even though every row is secondary

        #- targets is sorted ascending by TARGETID, NOT in first-appearance order
        #- (unlike desispec.util.ordered_unique); here TARGETID=300 appears first
        #- in the table but should come last in the output.
        t4 = Table()
        t4['TARGETID'] = [300, 100, 200]
        targets4, idx4 = find_target_priority(t4)
        self.assertTrue(np.all(targets4 == [100, 200, 300]))
        self.assertEqual(idx4[0], 1)  # TARGETID=100 -> row 1
        self.assertEqual(idx4[1], 2)  # TARGETID=200 -> row 2
        self.assertEqual(idx4[2], 0)  # TARGETID=300 -> row 0

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

    def test_create_summary_catalog_harmonizes_radec(self):
        """create_summary_catalog should propagate a single TARGET_RA/TARGET_DEC/
        REF_EPOCH/PMRA/PMDEC per TARGETID, preferring a primary target over a
        secondary one, and (among primaries) SURVEY=main over anything else.
        """
        def write_survey(indir, subdir, survey, program, scnd_col, rows):
            base = Table(rows=[r[0] for r in rows],
                         names=('TARGETID', 'TARGET_RA', 'TARGET_DEC', 'ZWARN',
                                'EFFTIME_SPEC', scnd_col))
            base.meta['SURVEY'] = survey
            base.meta['PROGRAM'] = program
            fn = f'{indir}/{subdir}/zpix-{survey}-{program}.fits'
            write_bintable(fn, base, extname='ZCATALOG', clobber=True)

            imaging = Table(rows=[r[1] for r in rows],
                             names=('TARGETID', 'REF_EPOCH', 'PMRA', 'PMDEC'))
            imaging.meta['SURVEY'] = survey
            imaging.meta['PROGRAM'] = program
            write_bintable(fn.replace('.fits', '-imaging.fits'), imaging,
                            extname='ZCATALOG_IMAGING', clobber=True)

            extra = Table()
            extra['TARGETID'] = base['TARGETID']
            extra.meta['SURVEY'] = survey
            extra.meta['PROGRAM'] = program
            write_bintable(fn.replace('.fits', '-extra.fits'), extra,
                            extname='ZCATALOG_EXTRA', clobber=True)

        with tempfile.TemporaryDirectory() as indir:
            os.makedirs(f'{indir}/main-dark')
            os.makedirs(f'{indir}/sv1-dark')

            #- main/dark: TARGETID 100 is PRIMARY (SCND_TARGET=0); TARGETID 200 unique to main
            write_survey(indir, 'main-dark', 'main', 'dark', 'SCND_TARGET', [
                ((100, 10.0, 1.0, 0, 500.0, 0), (100, 2015.5, 1.0, 2.0)),
                ((200, 50.0, 5.0, 0, 500.0, 0), (200, 2015.5, 0.0, 0.0)),
            ])

            #- sv1/dark: TARGETID 100 is SECONDARY (SV1_SCND_TARGET!=0), discrepant
            #- position, and *better* EFFTIME_SPEC so it would win ZCAT_PRIMARY
            write_survey(indir, 'sv1-dark', 'sv1', 'dark', 'SV1_SCND_TARGET', [
                ((100, 10.5, 1.5, 0, 1000.0, 7), (100, 2015.0, 0.0, 0.0)),
            ])

            create_summary_catalog('zpix', indir=indir, specprod='test')

            zall = Table.read(f'{indir}/zall/zall-pix-test.fits', hdu='ZCATALOG')
            imaging = Table.read(f'{indir}/zall/zall-pix-test-imaging.fits', hdu='ZCATALOG_IMAGING')

            is100 = zall['TARGETID'] == 100
            self.assertEqual(is100.sum(), 2)
            #- Both rows for TARGETID=100 should report the *main* row's position,
            #- even though the sv1 row has higher EFFTIME_SPEC (and thus is ZCAT_PRIMARY)
            self.assertTrue(np.all(zall['TARGET_RA'][is100] == 10.0))
            self.assertTrue(np.all(zall['TARGET_DEC'][is100] == 1.0))

            #- confirm ZCAT_PRIMARY did in fact pick the higher-EFFTIME_SPEC sv1 row,
            #- i.e. harmonization did not change which row is flagged primary
            primary_row = zall[is100 & zall['ZCAT_PRIMARY']]
            self.assertEqual(len(primary_row), 1)
            self.assertEqual(primary_row['SURVEY'][0].strip(), 'sv1')

            im100 = imaging['TARGETID'] == 100
            self.assertTrue(np.all(imaging['REF_EPOCH'][im100] == np.float32(2015.5)))
            self.assertTrue(np.all(imaging['PMRA'][im100] == 1.0))
            self.assertTrue(np.all(imaging['PMDEC'][im100] == 2.0))

            #- TARGETID=200 is untouched (only one row)
            is200 = zall['TARGETID'] == 200
            self.assertTrue(np.all(zall['TARGET_RA'][is200] == 50.0))
