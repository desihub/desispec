# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.scripts.qsoqn.
"""
import os
import tempfile
import unittest

import fitsio
import numpy as np

#- desispec.scripts.qsoqn imports quasarnp at module load time; skip these
#- tests (instead of erroring at collection) if quasarnp isn't installed.
try:
    from desispec.scripts.qsoqn import extract_redshift_info_from_RR
    qsoqn_available = True
except ImportError:
    qsoqn_available = False


@unittest.skipUnless(qsoqn_available, 'qsoqn (or dependencies) not available')
class TestQsoQN(unittest.TestCase):
    """Test desispec.scripts.qsoqn.
    """

    def setUp(self):
        #- Use a TemporaryDirectory object (cleaned up via .cleanup(), not
        #- shutil.rmtree on a separately cached path) to avoid ever removing
        #- the wrong directory; see desihub/desitarget#901.
        self._testdir_obj = tempfile.TemporaryDirectory()
        self.testdir = self._testdir_obj.name
        self.redrock_filename = os.path.join(self.testdir, 'redrock-test.fits')

    def tearDown(self):
        self._testdir_obj.cleanup()

    def _write_redrock(self, targetid, z):
        """Write a minimal REDSHIFTS HDU with the given TARGETID/Z columns."""
        data = [np.asarray(targetid), np.asarray(z)]
        names = ['TARGETID', 'Z']
        with fitsio.FITS(self.redrock_filename, 'rw', clobber=True) as fx:
            fx.write(data, names=names, extname='REDSHIFTS')

    def test_extract_redshift_info_from_RR_reorders(self):
        """Rows should be reordered to match the requested targetid order,
        even when the Redrock file stores them out of order or reversed."""
        redrock_tgid = np.array([30, 10, 20])
        redrock_z = np.array([3.0, 1.0, 2.0])
        self._write_redrock(redrock_tgid, redrock_z)

        targetid = np.array([10, 20, 30])
        result = extract_redshift_info_from_RR(self.redrock_filename, targetid)

        self.assertTrue(np.array_equal(result['TARGETID'], targetid))
        self.assertTrue(np.array_equal(result['Z'], [1.0, 2.0, 3.0]))

    def test_extract_redshift_info_from_RR_missing_targetid(self):
        """A requested targetid absent from the Redrock file should raise
        a clear error instead of a cryptic numpy TypeError."""
        self._write_redrock([10, 20], [1.0, 2.0])

        targetid = np.array([10, 20, 99])
        with self.assertRaises(ValueError):
            extract_redshift_info_from_RR(self.redrock_filename, targetid)

    def test_extract_redshift_info_from_RR_duplicate_targetid(self):
        """A duplicated targetid in the Redrock file should raise a clear
        error instead of a cryptic numpy TypeError."""
        self._write_redrock([10, 10, 20], [1.0, 1.1, 2.0])

        targetid = np.array([10, 20])
        with self.assertRaises(ValueError):
            extract_redshift_info_from_RR(self.redrock_filename, targetid)


if __name__ == '__main__':
    unittest.main()
