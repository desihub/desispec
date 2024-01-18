# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.database.
"""
from __future__ import absolute_import, division
import unittest, os
from datetime import datetime, timedelta
from shutil import rmtree

try:
    import sqlalchemy
    sqlalchemy_available = True
except ImportError:
    sqlalchemy_available = False


class TestDatabase(unittest.TestCase):
    """Test desispec.database.
    """
    @classmethod
    def setUpClass(cls):
        """Create unique test filename in a subdirectory.
        """
        # from uuid import uuid1
        # cls.testfile = 'test-{uuid}/test-{uuid}.fits'.format(uuid=uuid1())
        # cls.testyfile = 'test-{uuid}/test-{uuid}.yaml'.format(uuid=uuid1())
        # cls.testbrfile = 'test-{uuid}/test-br-{uuid}.fits'.format(uuid=uuid1())
        cls.testDir = os.path.join(os.environ['HOME'], 'desi_test_database')
        cls.origEnv = {'SPECPROD':None,
            "DESI_SPECTRO_DATA":None,
            "DESI_SPECTRO_REDUX":None}
        cls.testEnv = {'SPECPROD':'dailytest',
            "DESI_SPECTRO_DATA":os.path.join(cls.testDir,'spectro','data'),
            "DESI_SPECTRO_REDUX":os.path.join(cls.testDir,'spectro','redux')}
        for e in cls.origEnv:
            if e in os.environ:
                cls.origEnv[e] = os.environ[e]
            os.environ[e] = cls.testEnv[e]

    @classmethod
    def tearDownClass(cls):
        """Cleanup test files if they exist.
        """
        # for testfile in [cls.testfile, cls.testyfile, cls.testbrfile]:
        #     if os.path.exists(testfile):
        #         os.remove(testfile)
        #         testpath = os.path.normpath(os.path.dirname(testfile))
        #         if testpath != '.':
        #             os.removedirs(testpath)

        for e in cls.origEnv:
            if cls.origEnv[e] is None:
                del os.environ[e]
            else:
                os.environ[e] = cls.origEnv[e]

        if os.path.exists(cls.testDir):
            rmtree(cls.testDir)

    @unittest.skipUnless(sqlalchemy_available, "SQLAlchemy not installed; skipping datachallenge DB tests.")
    def test_datachallenge_classes(self):
        """Test SQLAlchemy classes in desispec.database.datachallenge.
        """
        from ..database.redshift import (Tile, Exposure, Frame, Fiberassign, Potential, Target)
        pass

    @unittest.skipUnless(sqlalchemy_available, "SQLAlchemy not installed; skipping metadata DB tests.")
    def test_metadata_classes(self):
        """Test SQLAlchemy classes in desispec.database.metadata.
        """
        from ..database.metadata import (utc, Base, FrameStatus, BrickStatus,
                                         Status, Night, ExposureFlavor)
        # self.assertIsNotNone(Base.metadata.tables)
        #
        # Simple ForeignKey tables.
        #
        st = Status(status='succeeded')
        self.assertEqual(str(st), "<Status(status='succeeded')>")
        ef = ExposureFlavor(flavor='science')
        self.assertEqual(str(ef), "<ExposureFlavor(flavor='science')>")
        ni = Night(night='20170101')
        self.assertEqual(str(ni), "<Night(night='20170101')>")
        #
        # Status tables.
        #
        fs = FrameStatus(id=1, frame_id=1, status='succeeded',
                         stamp=datetime(2017, 1, 1, 0, 0, 0, tzinfo=utc))
        self.assertEqual(str(fs), "<FrameStatus(id=1, frame_id=1, status='succeeded', stamp='2017-01-01 00:00:00+00:00')>")
        bs = BrickStatus(id=1, brick_id=1, status='succeeded',
                         stamp=datetime(2017, 1, 1, 0, 0, 0, tzinfo=utc))
        self.assertEqual(str(bs), "<BrickStatus(id=1, brick_id=1, status='succeeded', stamp='2017-01-01 00:00:00+00:00')>")

    def test_convert_dateobs(self):
        """Test desispec.database.util.convert_dateobs.
        """
        from pytz import utc
        from ..database.util import convert_dateobs
        ts = convert_dateobs('2019-01-03T01:11:33.247')
        self.assertEqual(ts.year, 2019)
        self.assertEqual(ts.month, 1)
        self.assertEqual(ts.microsecond, 247000)
        self.assertIsNone(ts.tzinfo)
        ts = convert_dateobs('2019-01-03T01:11:33.247', tzinfo=utc)
        self.assertEqual(ts.year, 2019)
        self.assertEqual(ts.month, 1)
        self.assertEqual(ts.microsecond, 247000)
        self.assertIs(ts.tzinfo, utc)

    def test_cameraid(self):
        """Test desispec.database.util.cameraid.
        """
        from ..database.util import cameraid
        self.assertEqual(cameraid('b0'), 0)
        self.assertEqual(cameraid('r5'), 15)
        self.assertEqual(cameraid('z9'), 29)

    def test_frameid(self):
        """Test desispec.database.util.frameid.
        """
        from ..database.util import frameid
        self.assertEqual(frameid(12345, 'b0'), 1234500)
        self.assertEqual(frameid(54321, 'r5'), 5432115)
        self.assertEqual(frameid(9876543, 'z9'), 987654329)


