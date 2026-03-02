"""
Test desispec.workflow.timing

These are primarily "does it run" test to catch API changes,
i.e. not "are they correct?" tests.
"""

import os, re
import unittest
import time
import tempfile
import json

from desispec.workflow import timing
from desiutil.timer import Timer

class TestWorkflowTiming(unittest.TestCase):

    def setUp(self):
        self.timingfile = os.path.join(tempfile.mkdtemp(), 'timing.json')

    def tearDown(self):
        if os.path.exists(self.timingfile):
            os.remove(self.timingfile)
        os.rmdir(os.path.dirname(self.timingfile))

    @classmethod
    def setUpClass(cls):
        cls.origTZ = os.getenv('TZ')

    @classmethod
    def tearDownClass(cls):
        if cls.origTZ is None:
            if 'TZ' in os.environ:
                del os.environ['TZ']
        else:
            os.environ['TZ'] = cls.origTZ

    def test_workflow_timing(self):
        night = timing.what_night_is_it()
        self.assertIsInstance(night, int)
        self.assertTrue(str(night).startswith('20'))
        self.assertEqual(len(str(night)), 8)

        start_time = timing.get_nightly_start_time()
        self.assertGreaterEqual(start_time, 0)
        self.assertLess(start_time, 24)

        end_time = timing.get_nightly_end_time()
        self.assertGreaterEqual(end_time, 0)
        self.assertLess(end_time, 24)

        os.environ['TZ'] = 'US/California'
        timing.ensure_tucson_time()
        self.assertEqual(os.environ['TZ'], 'US/Arizona')
        del os.environ['TZ']
        timing.ensure_tucson_time()
        self.assertEqual(os.environ['TZ'], 'US/Arizona')

        timestr = timing.nersc_format_datetime()
        timestr = timing.nersc_end_time()
        yesno = timing.during_operating_hours()

        # NOT Tested: timing.wait_for_cals because that has time-dependent behavior
        # and long sleeps if it doesn't find the files it is looking for.

    def test_log_timer(self):
        """
        Test desispec.workflow.timeing.log_timer
        """

        # ensure that timingfile doesn't already exist
        self.assertFalse(os.path.exists(self.timingfile))

        # create new timer and save to timing file
        t1 = Timer()
        t1.start('blat')
        time.sleep(0.01)
        t1.stop('blat')
        t1.start('foo')
        time.sleep(0.02)
        t1.stop('foo')
        timing.log_timer(t1, self.timingfile)

        self.assertTrue(os.path.exists(self.timingfile))
        with open(self.timingfile) as f:
            content = json.load(f)

        self.assertIn('blat', content.keys())
        self.assertIn('foo', content.keys())

        # create second timer and augment timing file
        t1 = Timer()
        t1.start('blat')
        time.sleep(0.01)
        t1.stop('blat')
        t1.start('bar')
        time.sleep(0.02)
        t1.stop('bar')
        timing.log_timer(t1, self.timingfile)

        # check updated contents
        with open(self.timingfile) as f:
            content = json.load(f)

        self.assertIn('blat', content.keys())
        self.assertIn('blat.1', content.keys())
        self.assertIn('foo', content.keys())
        self.assertIn('bar', content.keys())
        self.assertNotIn('foo.1', content.keys())
        self.assertNotIn('bar.1', content.keys())

        # third round, again with blat
        t1 = Timer()
        t1.start('blat')
        time.sleep(0.01)
        t1.stop('blat')
        t1.start('bar')
        time.sleep(0.02)
        t1.stop('bar')
        t1.start('biz')
        time.sleep(0.01)
        t1.stop('biz')
        timing.log_timer(t1, self.timingfile)

        # check updated contents
        with open(self.timingfile) as f:
            content = json.load(f)

        self.assertIn('blat', content.keys())
        self.assertIn('blat.1', content.keys())
        self.assertIn('blat.2', content.keys())
        self.assertIn('foo', content.keys())
        self.assertIn('biz', content.keys())
        self.assertIn('bar', content.keys())
        self.assertIn('bar.1', content.keys())

