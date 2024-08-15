# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.io.
"""

import unittest
import os
import shutil
from importlib import resources
import tempfile

from desispec.calibfinder import CalibFinder

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

        #- without an entry in DESI_SPECTRO_DARK, even creating the CalibFinder fails
        with self.assertRaises(OSError):
            cfinder = CalibFinder([phdr,camhdr])

        #- but fallback option should work
        cfinder = CalibFinder([phdr,camhdr], fallback_on_dark_not_found=True)
        darkfile = cfinder.findfile('DARK')
        self.assertTrue(darkfile is not None)
