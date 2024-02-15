# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.io.fibermap.
"""
import os
import tempfile
import unittest
from unittest.mock import patch, call
import warnings
from shutil import rmtree

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from astropy.io import fits
from astropy.table import Table, Column
import fitsio
from desiutil.annotate import check_comment_length

from ..io.fibermap import (fibermap_columns, empty_fibermap, write_fibermap, read_fibermap,
                           find_fiberassign_file, update_survey_keywords,
                           assemble_fibermap, annotate_fibermap)


standard_nersc_environment = ('NERSC_HOST' in os.environ and
                              os.getenv('DESI_SPECTRO_DATA') == '/global/cfs/cdirs/desi/spectro/data')


class TestIOFibermap(unittest.TestCase):
    """Test desispec.io.fibermap.
    """

    @classmethod
    def setUpClass(cls):
        """Create unique test filename in a subdirectory.
        """
        cls.testDir = tempfile.mkdtemp()
        # cls.readonlyDir = tempfile.mkdtemp()
        cls.testfile = os.path.join(cls.testDir, 'desispec_test_io_fibermap.fits')
        # cls.testlog = os.path.join(cls.testDir, 'desispec_test_io.log')
        cls.origEnv = {'SPECPROD': None,
                       "DESI_ROOT": None,
                       "DESI_ROOT_READONLY": None,
                       "DESI_SPECTRO_DATA": None,
                       "DESI_SPECTRO_REDUX": None,
                       "DESI_SPECTRO_CALIB": None,
                       }
        # cls.testEnv = {'SPECPROD':'dailytest',
        #                "DESI_ROOT": cls.testDir,
        #                "DESI_ROOT_READONLY": cls.readonlyDir,
        #                "DESI_SPECTRO_DATA": os.path.join(cls.testDir, 'spectro', 'data'),
        #                "DESI_SPECTRO_REDUX": os.path.join(cls.testDir, 'spectro', 'redux'),
        #                "DESI_SPECTRO_CALIB": os.path.join(cls.testDir, 'spectro', 'calib'),
        #                }
        cls.dataDir = os.path.join(cls.testDir, 'spectro', 'data')
        # cls.datadir = cls.testEnv['DESI_SPECTRO_DATA']
        # cls.reduxdir = os.path.join(cls.testEnv['DESI_SPECTRO_REDUX'],
        #                             cls.testEnv['SPECPROD'])
        # for e in cls.origEnv:
        #     if e in os.environ:
        #         cls.origEnv[e] = os.environ[e]
        #     os.environ[e] = cls.testEnv[e]

    @classmethod
    def tearDownClass(cls):
        """Cleanup test files if they exist.
        """
        # for testfile in [cls.testfile, cls.testlog]:
        #     if os.path.exists(testfile):
        #         os.remove(testfile)
        # for e in cls.origEnv:
        #     if cls.origEnv[e] is None:
        #         del os.environ[e]
        #     else:
        #         os.environ[e] = cls.origEnv[e]

        if os.path.isdir(cls.testDir):
            rmtree(cls.testDir)

        # reset the readonly cache
        # from ..io import meta
        # meta._desi_root_readonly = None

    # def setUp(self):
    #     if os.path.isdir(self.datadir):
    #         rmtree(self.datadir)
    #     if os.path.isdir(self.reduxdir):
    #         rmtree(self.reduxdir)

    def tearDown(self):
        for testfile in [self.testfile]:
            if os.path.exists(testfile):
                os.remove(testfile)

        # restore environment variables if test changed them
        # for key, value in self.testEnv.items():
        #     os.environ[key] = value

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

        self.assertListEqual(fm3.colnames, [c[0] for c in fibermap_columns if c[4] == 'empty'])

        fm4 = empty_fibermap(10, specmin=495, survey='cmx')
        self.assertIn('CMX_TARGET', fm4.colnames)
        self.assertEqual(fm4.meta['SURVEY'], 'cmx')

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
        nightdir = os.path.join(self.dataDir, str(night))
        os.makedirs(nightdir)
        os.makedirs(f'{nightdir}/00012345')
        os.makedirs(f'{nightdir}/00012346')
        os.makedirs(f'{nightdir}/00012347')
        os.makedirs(f'{nightdir}/00012348')
        fafile = f'{nightdir}/00012346/fiberassign-001111.fits'
        fafilegz = f'{nightdir}/00012347/fiberassign-001122.fits'

        fx = open(fafile, 'w'); fx.close()
        fx = open(fafilegz, 'w'); fx.close()
        with patch.dict('os.environ', {'DESI_SPECTRO_DATA': self.dataDir}):

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
        fibermap_comments = dict([(row[0], row[3]) for row in fibermap_columns])
        n_long = check_comment_length(fibermap_comments, error=False)
        self.assertEqual(n_long, 0)

    def test_update_survey_keywords(self):
        """Test updating survey keywords.
        """
        #- Standard case with all the keywords - no changes
        hdr = dict(TILEID=1, SURVEY='dark', FAPRGRM='main', FAFLAVOR='maindark')
        hdr0 = hdr.copy()
        update_survey_keywords(hdr)
        self.assertEqual(hdr, hdr0)

        #- Tiles 63501-63505 only have TILEID
        hdr = dict(TILEID=63501)
        update_survey_keywords(hdr)
        self.assertEqual(hdr['SURVEY'], 'cmx')
        self.assertEqual(hdr['FAPRGRM'], 'bright')
        self.assertEqual(hdr['FAFLAVOR'], 'cmxbright')

        #- Other cases with TILEID but unknown details
        hdr = dict(TILEID=70004)
        update_survey_keywords(hdr)
        self.assertEqual(hdr['SURVEY'], 'cmx')
        self.assertEqual(hdr['FAPRGRM'], 'unknown')
        self.assertEqual(hdr['FAFLAVOR'], 'cmxunknown')

        #- TILEID and FAFLAVOR but not SURVEY or FAPRGRM
        hdr = dict(TILEID=80220, FAFLAVOR='dithlost')
        update_survey_keywords(hdr)
        self.assertEqual(hdr['SURVEY'], 'cmx')
        self.assertEqual(hdr['FAPRGRM'], 'dithlost')

        hdr = dict(TILEID=80605, FAFLAVOR='cmxlrgqso')
        update_survey_keywords(hdr)
        self.assertEqual(hdr['SURVEY'], 'sv1')  # yes, sv1 not cmx
        self.assertEqual(hdr['FAPRGRM'], 'lrgqso')

        hdr = dict(TILEID=80606, FAFLAVOR='cmxelg')
        update_survey_keywords(hdr)
        self.assertEqual(hdr['SURVEY'], 'sv1')  # yes, sv1 not cmx
        self.assertEqual(hdr['FAPRGRM'], 'elg')

        hdr = dict(TILEID=80611, FAFLAVOR='sv1bgsmws')
        update_survey_keywords(hdr)
        self.assertEqual(hdr['SURVEY'], 'sv1')
        self.assertEqual(hdr['FAPRGRM'], 'bgsmws')

        #- TILEID, SURVEY, and FAFLAVOR but not FAPRGRM
        hdr = dict(TILEID=81000, SURVEY='sv2', FAFLAVOR='sv2dark')
        update_survey_keywords(hdr)
        self.assertEqual(hdr['SURVEY'], 'sv2')
        self.assertEqual(hdr['FAPRGRM'], 'dark')

    @unittest.skipUnless(standard_nersc_environment, "not at NERSC")
    def test_assemble_fibermap(self):
        """Test creation of fibermaps from raw inputs"""
        for night, expid in [
            (20200219, 51039),  #- old SPS header
            (20200315, 55611),  #- new SPEC header
            ]:
            print(f'Creating fibermap for {night}/{expid}')
            fm = assemble_fibermap(night, expid)['FIBERMAP'].data

            #- unmatched positioners aren't in coords files and have
            #- FIBER_X/Y == 0, but most should be non-zero
            self.assertLess(np.count_nonzero(fm['FIBER_X'] == 0.0), 50)
            self.assertLess(np.count_nonzero(fm['FIBER_Y'] == 0.0), 50)

            #- all with FIBER_X/Y == 0 should have a FIBERSTATUS flag
            ii = (fm['FIBER_X'] == 0.0) & (fm['FIBER_Y'] == 0.0)
            self.assertTrue(np.all(fm['FIBERSTATUS'][ii] != 0))

            #- and platemaker x/y shouldn't match fiberassign x/y
            self.assertTrue(np.all(fm['FIBER_X'] != fm['FIBERASSIGN_X']))
            self.assertTrue(np.all(fm['FIBER_Y'] != fm['FIBERASSIGN_Y']))

            #- spot check existence of a few other columns
            for col in (
                'TARGETID', 'LOCATION', 'FIBER', 'TARGET_RA', 'TARGET_DEC',
                'PLATE_RA', 'PLATE_DEC',
                ):
                self.assertIn(col, fm.columns.names)

    @unittest.skipUnless(standard_nersc_environment, "not at NERSC")
    def test_compare_empty_to_assemble(self):
        """Compare the output of empty_fibermap to assemble_fibermap.
        """
        empty = fits.convenience.table_to_hdu(empty_fibermap(5000))
        fm_hdu = assemble_fibermap(20210517, 89031)
        self.assertListEqual(empty.data.columns.names, fm_hdu.data.columns.names)

    @unittest.skipUnless(standard_nersc_environment, "not at NERSC")
    def test_missing_input_files(self):
        """Test creation of fibermaps with missing input files"""
        #- missing coordinates file for this exposure
        night, expid = 20200219, 51053
        with self.assertRaises(FileNotFoundError):
            fm = assemble_fibermap(night, expid)

        #- But should work with force=True
        fm = assemble_fibermap(night, expid, force=True)

        #- ...albeit with FIBER_X/Y == 0
        assert np.all(fm['FIBERMAP'].data['FIBER_X'] == 0.0)
        assert np.all(fm['FIBERMAP'].data['FIBER_Y'] == 0.0)

    @unittest.skipUnless(standard_nersc_environment, "not at NERSC")
    def test_missing_input_columns(self):
        """Test creation of fibermaps with missing input columns"""
        #- second exposure of split, missing fiber location information
        #- in coordinates file, but info is present in previous exposure
        #- that was same tile and first in sequence
        fm1 = assemble_fibermap(20210406, 83714)['FIBERMAP'].data
        fm2 = assemble_fibermap(20210406, 83715)['FIBERMAP'].data

        def nanequal(a, b):
            """Compare two arrays treating NaN==NaN"""
            return np.equal(a, b, where=~np.isnan(a))

        assert np.all(nanequal(fm1['FIBER_X'], fm2['FIBER_X']))
        assert np.all(nanequal(fm1['FIBER_Y'], fm2['FIBER_Y']))
        assert np.all(nanequal(fm1['FIBER_RA'], fm2['FIBER_RA']))
        assert np.all(nanequal(fm1['FIBER_DEC'], fm2['FIBER_DEC']))
        assert np.all(nanequal(fm1['DELTA_X'], fm2['DELTA_X']))
        assert np.all(nanequal(fm1['DELTA_Y'], fm2['DELTA_Y']))

    @patch('desispec.io.fibermap.get_logger')
    def test_annotate_fibermap(self, mock_log):
        """Test updating units and column descriptions in a fibermap HDU.
        """
        with self.assertRaises(ValueError) as e:
            fibermap = annotate_fibermap(dict())
        #
        # Test normal operation
        #
        fibermap = empty_fibermap(500)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*nmgy.*", category=AstropyUserWarning)
            fibermap_hdu = fits.BinTableHDU(fibermap)
        fibermap_hdu = annotate_fibermap(fibermap_hdu)
        self.assertEqual(fibermap_hdu.header.comments['TTYPE7'], 'Barycentric right ascension in ICRS')   # TARGET_RA
        self.assertEqual(fibermap_hdu.header['TUNIT7'], 'deg')  # TARGET_RA
        mock_log().error.assert_not_called()
        #
        # This is a by-product of creating the table via empty_fibermap.
        #
        mock_log().warning.assert_has_calls([call("Overriding units for column '%s': '%s' -> '%s'.", 'PMRA', 'mas yr-1', 'mas yr^-1'),
                                             call("Overriding units for column '%s': '%s' -> '%s'.", 'PMDEC', 'mas yr-1', 'mas yr^-1')])
        #
        # Test with unexpected column
        #
        fibermap = empty_fibermap(500)
        unex = Column(np.arange(500), name='UNEXPECTED')
        fibermap.add_column(unex, index=2)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*nmgy.*", category=AstropyUserWarning)
            fibermap_hdu = fits.BinTableHDU(fibermap)
        fibermap_hdu = annotate_fibermap(fibermap_hdu)
        self.assertEqual(fibermap_hdu.header['TTYPE3'], 'UNEXPECTED')
        self.assertEqual(fibermap_hdu.header.comments['TTYPE3'], '')
        mock_log().error.assert_called_once_with('Unexpected column name, %s, found in fibermap HDU! Annotation will be skipped on this column.', 'UNEXPECTED')
