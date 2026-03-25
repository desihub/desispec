"""
Test desispec.workflow.batch
"""

import os
import re
import math
import shutil
import tempfile
import unittest

import numpy as np
from astropy.table import Table, vstack

from desispec.workflow import batch

class TestWorkflowBatch(unittest.TestCase):
    """Test desispec.workflow.calibration_selection
    """

    #- Tests might alter NERSC host so reset after each test
    @classmethod
    def setUpClass(cls):
        cls._cached_nersc_host = os.getenv('NERSC_HOST')  # None if not set
        cls._cached_redux = os.getenv('DESI_SPECTRO_REDUX')
        cls._cached_specprod = os.getenv('SPECPROD')
        cls._tmpdir = tempfile.mkdtemp()
        os.environ['DESI_SPECTRO_REDUX'] = cls._tmpdir
        os.environ['SPECPROD'] = 'test'

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)
        if cls._cached_redux is None:
            os.environ.pop('DESI_SPECTRO_REDUX', None)
        else:
            os.environ['DESI_SPECTRO_REDUX'] = cls._cached_redux
        if cls._cached_specprod is None:
            os.environ.pop('SPECPROD', None)
        else:
            os.environ['SPECPROD'] = cls._cached_specprod

    def tearDown(self):
        if self._cached_nersc_host is None:
            if 'NERSC_HOST' in os.environ:
                del os.environ['NERSC_HOST']
            else:
                pass
        else:
            os.environ['NERSC_HOST'] = self._cached_nersc_host

    def test_parse_reservation(self):
        """Test parse_reservation at NERSC"""
        os.environ['NERSC_HOST'] = 'perlmutter'

        # single reservation -> same name regardless of CPU or GPU
        self.assertEqual(batch.parse_reservation('blat', 'arc'),       'blat')
        self.assertEqual(batch.parse_reservation('blat', 'ccdcalib'),  'blat')
        self.assertEqual(batch.parse_reservation('blat', 'flat'),      'blat')
        self.assertEqual(batch.parse_reservation('blat', 'tilenight'), 'blat')
        self.assertEqual(batch.parse_reservation('blat', 'ztile'),     'blat')

        # blat,foo -> cpu_reservation, gpu_reservtion
        self.assertEqual(batch.parse_reservation('blat,foo', 'arc'),       'blat')
        self.assertEqual(batch.parse_reservation('blat,foo', 'ccdcalib'),  'blat')
        self.assertEqual(batch.parse_reservation('blat,foo', 'flat'),      'foo')
        self.assertEqual(batch.parse_reservation('blat,foo', 'tilenight'), 'foo')
        self.assertEqual(batch.parse_reservation('blat,foo', 'ztile'),     'foo')

        # "none" special cases to no reservation
        self.assertEqual(batch.parse_reservation('blat,none', 'arc'),       'blat')
        self.assertEqual(batch.parse_reservation('blat,none', 'ccdcalib'),  'blat')
        self.assertEqual(batch.parse_reservation('blat,none', 'flat'),      None)
        self.assertEqual(batch.parse_reservation('blat,none', 'tilenight'), None)
        self.assertEqual(batch.parse_reservation('blat,none', 'ztile'),     None)

        self.assertEqual(batch.parse_reservation('none,foo', 'arc'),       None)
        self.assertEqual(batch.parse_reservation('none,foo', 'ccdcalib'),  None)
        self.assertEqual(batch.parse_reservation('none,foo', 'flat'),      'foo')
        self.assertEqual(batch.parse_reservation('none,foo', 'tilenight'), 'foo')
        self.assertEqual(batch.parse_reservation('none,foo', 'ztile'),     'foo')

        # And None -> None
        self.assertEqual(batch.parse_reservation(None, 'arc'),       None)
        self.assertEqual(batch.parse_reservation(None, 'ccdcalib'),  None)
        self.assertEqual(batch.parse_reservation(None, 'flat'),      None)
        self.assertEqual(batch.parse_reservation(None, 'tilenight'), None)
        self.assertEqual(batch.parse_reservation(None, 'ztile'),     None)

        # test errors
        with self.assertRaises(ValueError):
            batch.parse_reservation('blat,foo,bar', 'arc')

    def test_ccdcalib_darknight_capping(self):
        """Test that darknight srun -n is capped and runtime scaled when ncameras > 10 per node."""
        from desispec.workflow.batch_writer import create_ccdcalib_batch_script
        from desispec.workflow.batch import determine_resources

        system_name = 'perlmutter-gpu'
        night = 20250101
        expids = [11111]

        # a0123 = 12 cameras (4 spectrographs * 3 arms), which exceeds 10 per node
        camword_many = 'a0123'
        ntasks, nodes, base_runtime = determine_resources(
            12, 'ccdcalib', queue='regular', nexps=1, system_name=system_name)
        self.assertGreater(float(ntasks) / float(nodes), 10,
                           'Test prerequisite: ntasks/nodes must exceed 10 for capping to apply')

        script = create_ccdcalib_batch_script(
            night=night, expids=expids, camword=camword_many,
            do_darknight=True, queue='regular', system_name=system_name)

        with open(script) as f:
            content = f.read()

        # Verify that the darknight srun line uses the capped number of tasks (10 per node)
        max_ranks_per_node = 10
        expected_dn_ntasks = max_ranks_per_node * nodes
        srun_pattern = re.compile(
            r'srun\s+-N\s+\d+\s+-n\s+(\d+)\s+.*desi_compute_dark_night')
        match = srun_pattern.search(content)
        self.assertIsNotNone(match, 'darknight srun line not found in script')
        actual_n = int(match.group(1))
        self.assertEqual(actual_n, expected_dn_ntasks,
                         f'Expected darknight -n={expected_dn_ntasks} (capped), got {actual_n}')

        # Verify the walltime includes the additional time from looping
        loops = math.ceil(float(ntasks) / float(expected_dn_ntasks))
        total_runtime = base_runtime + 10.0 * loops
        expected_hh = int(total_runtime // 60)
        expected_mm = int(total_runtime % 60)
        expected_time = f'#SBATCH --time={expected_hh:02d}:{expected_mm:02d}:00'
        self.assertIn(expected_time, content,
                      f'Expected walltime "{expected_time}" not found in script')

    def test_ccdcalib_darknight_no_capping(self):
        """Test that darknight srun -n is not capped when ncameras <= 10 per node."""
        from desispec.workflow.batch_writer import create_ccdcalib_batch_script
        from desispec.workflow.batch import determine_resources

        system_name = 'perlmutter-gpu'
        night = 20250101
        expids = [22222]

        # a012 = 9 cameras (3 spectrographs * 3 arms), which does not exceed 10 per node
        camword_few = 'a012'
        ntasks, nodes, base_runtime = determine_resources(
            9, 'ccdcalib', queue='regular', nexps=1, system_name=system_name)
        self.assertLessEqual(float(ntasks) / float(nodes), 10,
                             'Test prerequisite: ntasks/nodes must not exceed 10 for no-capping case')

        script = create_ccdcalib_batch_script(
            night=night, expids=expids, camword=camword_few,
            do_darknight=True, queue='regular', system_name=system_name)

        with open(script) as f:
            content = f.read()

        # With no capping, the srun -n should equal ntasks
        srun_pattern = re.compile(
            r'srun\s+-N\s+\d+\s+-n\s+(\d+)\s+.*desi_compute_dark_night')
        match = srun_pattern.search(content)
        self.assertIsNotNone(match, 'darknight srun line not found in script')
        actual_n = int(match.group(1))
        self.assertEqual(actual_n, ntasks,
                         f'Expected darknight -n={ntasks} (no cap), got {actual_n}')

        # Runtime should include exactly one loop (all tasks fit within max_ranks_per_node)
        total_runtime = base_runtime + 10.0
        expected_hh = int(total_runtime // 60)
        expected_mm = int(total_runtime % 60)
        expected_time = f'#SBATCH --time={expected_hh:02d}:{expected_mm:02d}:00'
        self.assertIn(expected_time, content,
                      f'Expected walltime "{expected_time}" not found in script')

