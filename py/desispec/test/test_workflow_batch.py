"""
Test desispec.workflow.batch
"""

import os
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

    def test_max_nodes_for_jobdesc(self):
        """PSFNIGHT gets a larger node cap than everything else"""
        self.assertEqual(batch.max_nodes_for_jobdesc('PSFNIGHT'), 15)
        self.assertEqual(batch.max_nodes_for_jobdesc('psfnight'), 15)
        for jobdesc in ('NIGHTLYFLAT', 'CTEFLAT', 'ARC', 'FLAT', 'TILENIGHT'):
            self.assertEqual(batch.max_nodes_for_jobdesc(jobdesc), 5)

    def test_determine_resources_cteflat(self):
        """CTEFLAT is a valid jobdesc giving the same resources as FLAT"""
        os.environ['NERSC_HOST'] = 'perlmutter'
        flat = batch.determine_resources(30, 'FLAT', system_name='perlmutter-gpu')
        cteflat = batch.determine_resources(30, 'CTEFLAT', system_name='perlmutter-gpu')
        self.assertEqual(flat, cteflat)

        # and an unknown jobdesc still raises
        with self.assertRaises(ValueError):
            batch.determine_resources(30, 'NOTAJOB', system_name='perlmutter-gpu')

    def test_determine_resources_node_cap(self):
        """The node cap applied by determine_resources comes from the helper"""
        os.environ['NERSC_HOST'] = 'perlmutter'
        for jobdesc in ('ARC', 'FLAT', 'PSFNIGHT', 'NIGHTLYFLAT', 'CTEFLAT'):
            ncores, nodes, runtime = batch.determine_resources(
                    30, jobdesc, system_name='perlmutter-cpu')
            self.assertLessEqual(nodes, batch.max_nodes_for_jobdesc(jobdesc))

