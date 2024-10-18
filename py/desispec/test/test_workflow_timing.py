"""
Test desispec.workflow.timing

These are primarily "does it run" test to catch API changes,
i.e. not "are they correct?" tests.
"""

import os, re
import unittest

from desispec.workflow import timing

class TestWorkflowTiming(unittest.TestCase):

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



