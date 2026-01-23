# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.io.
"""

import unittest
import os
import shutil
from importlib import resources
import tempfile
from astropy.table import Table

from ..calibfinder import CalibFinder

_standard_calib_dirs = ('DESI_SPECTRO_CALIB' in os.environ) and ('DESI_SPECTRO_DARK' in os.environ)

class TestCalibFinder(unittest.TestCase):
    """Test desispec.calibfinder
    """

    @classmethod
    def setUpClass(cls):

        #- Cache original environment
        cls.origenv = dict()
        for key in ['DESI_SPECTRO_CALIB', 'DESI_SPECTRO_DARK']:
            cls.origenv[key] = os.getenv(key, None)

        #- Prepare alternate $DESI_SPECTRO_CALIB for testing
        cls.calibdir = tempfile.mkdtemp()
        specdir = os.path.join(cls.calibdir,"spec/sp0")
        os.makedirs(specdir)
        for c in "brz" :
            shutil.copy(str(resources.files('desispec').joinpath(f'test/data/ql/{c}0.yaml')), os.path.join(specdir,f"{c}0.yaml"))

        cls.test_flaggedfile = os.path.join(cls.calibdir, 'test_flagged_fibers.ecsv')

        table = Table()
        table['EXPID'] = [12345, 12345, 67890, 99999, 4680]
        table['FIBERS'] = ['0:5', '100,101,102', '4995:5000', '2500', '100-104']
        table['FIBERSTATUS_BITNAME'] = ['BRIGHTNEIGHBOR', 'BADFIBER', 'RESERVED31', 'BADTRACE', 'BRIGHTNEIGHBOR|BADFIBER']
        #                 maskval      [ 2048,             65536,      2147483648,   131072,    2048+65536   ]
        #                 bitnum       [ 11,               16,         31,           17,        11;16       ]
        table['COMMENTS'] = ['Test multiple rows same EXPID', 'Test multiple rows same EXPID',
                             'High Mask Bit', 'Test single fiber', 'Test multiple bits']

        table.write(cls.test_flaggedfile, format='ascii.ecsv', overwrite=True)

    @classmethod
    def tearDownClass(cls):
        #- remove temporary calibration directory
        if cls.calibdir.startswith(tempfile.gettempdir()) and os.path.isdir(cls.calibdir) :
            shutil.rmtree(cls.calibdir)

    def tearDown(self):
        #- restore original environment after every test;
        #- some tests use default env; others use alternate $DESI_SPECTRO_CALIB
        for key, value in self.origenv.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

    def test_init(self):
        """Test basic initialization using test $DESI_SPECTRO_CALIB
        """
        os.environ["DESI_SPECTRO_CALIB"] = self.calibdir
        pheader={"DATE-OBS":'2018-11-30T12:42:10.442593-05:00',"DOSVER":'SIM'}
        header={"DETECTOR":'SIM',"CAMERA":'b0      ',"FEEVER":'SIM'}
        cfinder = CalibFinder([pheader,header])
        print(cfinder.value("DETECTOR"))
        if cfinder.haskey("BIAS") :
            print(cfinder.findfile("BIAS"))

    @unittest.skipIf(not _standard_calib_dirs, "$DESI_SPECTRO_CALIB or $DESI_SPECTRO_DARK not set")
    def test_missing_darks(self):
        """Missing dark files is only fatal if darks are requested
        """

        #- Commissioning era data from 20200219 expid 51053, for which we don't have
        #- darks in $DESI_SPECTRO_DARK
        phdr = {"DATE-OBS":"2020-02-20T08:59:59.104576", "DOSVER":"trunk"}
        camhdr = {"DETECTOR":"sn22797", "CAMERA":"b0", "FEEVER":"v20160312", "SPECID":4,
                  "CCDCFG":"default_sta_20190717.cfg",
                  "CCDTMING":"default_sta_timing_20180905.txt"}

        #- without an entry in DESI_SPECTRO_DARK, creating the CalibFinder works but requesting dark fails
        cfinder = CalibFinder([phdr,camhdr])
        with self.assertRaises(KeyError):
            cfinder.findfile('DARK')

        #- but fallback option should work
        cfinder = CalibFinder([phdr,camhdr], fallback_on_dark_not_found=True)
        darkfile = cfinder.findfile('DARK')
        self.assertTrue(darkfile is not None)

    def _set_env_calibdir(self):
        """Set $DESI_SPECTRO_CALIB to the test calibdir
        """
        reset_calib_env = False
        if os.getenv('DESI_SPECTRO_CALIB', None) is None:
            os.environ['DESI_SPECTRO_CALIB'] = self.calibdir
            reset_calib_env = True
        return reset_calib_env

    def _remove_env_calibdir(self, reset_calib_env):
        """Set $DESI_SPECTRO_CALIB to the test calibdir

        Args:
            reset_calib_env (bool): if True, remove the env var to reset to original state
        """
        if reset_calib_env:
            del os.environ['DESI_SPECTRO_CALIB']

    def test_flaggedfiber_single_row_range(self):
        """Test parsing a fiber range from a single matching row."""
        from ..calibfinder import get_flagged_fibers
        reset_calib_env = self._set_env_calibdir()
        fibers, masks = get_flagged_fibers(67890, filename=self.test_flaggedfile)

        expected_fibers = [4995, 4996, 4997, 4998, 4999]
        expected_masks = [2147483648] * 5

        self.assertEqual(fibers, expected_fibers)
        self.assertEqual(masks, expected_masks)
        self.assertEqual(len(fibers), len(masks))
        self._remove_env_calibdir(reset_calib_env)


    def test_flaggedfiber_multiple_rows_same_expid(self):
        """Test combining fibers from multiple rows with the same EXPID."""
        from ..calibfinder import get_flagged_fibers
        reset_calib_env = self._set_env_calibdir()
        fibers, masks = get_flagged_fibers(12345, filename=self.test_flaggedfile)

        expected_fibers = [0, 1, 2, 3, 4, 100, 101, 102]
        expected_masks = [2048, 2048, 2048, 2048, 2048, 65536, 65536, 65536]

        self.assertEqual(fibers, expected_fibers)
        self.assertEqual(masks, expected_masks)
        self.assertEqual(len(fibers), len(masks))
        self._remove_env_calibdir(reset_calib_env)

    def test_flaggedfiber_single_fiber(self):
        """Test parsing a single fiber number."""
        from ..calibfinder import get_flagged_fibers
        reset_calib_env = self._set_env_calibdir()
        fibers, masks = get_flagged_fibers(99999, filename=self.test_flaggedfile)

        self.assertEqual(fibers, [2500])
        self.assertEqual(masks, [131072])
        self._remove_env_calibdir(reset_calib_env)

    def test_flaggedfiber_nonexistent_expid(self):
        """Test that nonexistent EXPID returns empty lists."""
        from ..calibfinder import get_flagged_fibers
        reset_calib_env = self._set_env_calibdir()
        fibers, masks = get_flagged_fibers(11111, filename=self.test_flaggedfile)

        self.assertEqual(fibers, [])
        self.assertEqual(masks, [])
        self._remove_env_calibdir(reset_calib_env)

    def test_flaggedfiber_mask_is_single_bit(self):
        """Test that each mask value has only a single bit set."""
        from ..calibfinder import get_flagged_fibers
        reset_calib_env = self._set_env_calibdir()
        fibers, masks = get_flagged_fibers(12345, filename=self.test_flaggedfile)

        for mask in masks:
            self.assertGreater(mask, 0)
            self.assertEqual(mask & (mask - 1), 0, f"Mask {mask} has more than one bit set")
        self._remove_env_calibdir(reset_calib_env)

    def test_flaggedfiber_fibers_in_valid_range(self):
        """Test that all fiber numbers are in valid range 0-4999."""
        from ..calibfinder import get_flagged_fibers
        reset_calib_env = self._set_env_calibdir()
        for expid in [12345, 67890, 99999]:
            fibers, masks = get_flagged_fibers(expid, filename=self.test_flaggedfile)
            for fiber in fibers:
                self.assertGreaterEqual(fiber, 0)
                self.assertLessEqual(fiber, 4999)
        self._remove_env_calibdir(reset_calib_env)

    def test_flaggedfiber_filename_none_raises(self):
        """Test that None filename returns empty lists."""
        from ..calibfinder import get_flagged_fibers
        reset_calib_env = self._set_env_calibdir()
        fibers, masks = get_flagged_fibers(12345, filename=None)
        self.assertEqual(len(fibers), 0)
        self.assertEqual(len(masks), 0)
        self._remove_env_calibdir(reset_calib_env)

    def test_flaggedfiber_highest_bit_mask(self):
        """Test handling of highest bit in 32-bit mask (bit 31)."""
        from ..calibfinder import get_flagged_fibers
        reset_calib_env = self._set_env_calibdir()
        fibers, masks = get_flagged_fibers(67890, filename=self.test_flaggedfile)

        self.assertEqual(masks[0], 2147483648)
        self.assertEqual(masks[0], 1 << 31)
        self._remove_env_calibdir(reset_calib_env)

        max_32bit = 2**32 - 1
        for mask in masks:
            self.assertLessEqual(mask, max_32bit)
            self.assertGreaterEqual(mask, 0)
        self._remove_env_calibdir(reset_calib_env)

    def test_flaggedfiber_multiplebits(self):
        """Test handling of multiple bits in a single table row."""
        from ..calibfinder import get_flagged_fibers
        reset_calib_env = self._set_env_calibdir()
        fibers, masks = get_flagged_fibers(4680, filename=self.test_flaggedfile)
        self.assertEqual(len(masks), 4)
        for mask in masks:
            self.assertEqual(mask, 2048+65536)
        self._remove_env_calibdir(reset_calib_env)
