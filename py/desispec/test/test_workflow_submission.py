"""
Test cross-night dependency logic in desispec.workflow.submission.submit_linkcal_jobs.

Regression tests for the linkcal refnight dependency feature: when a refnight
proctable with in-flight calib jobs is present, the new-night linkcal prow must
acquire INT_DEP_IDS / LATEST_DEP_QID pointing at those jobs.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from desispec.io.meta import findfile
from desispec.workflow.proctable import (
    default_prow,
    get_default_qid,
    instantiate_processing_table,
)
from desispec.workflow.tableio import write_table
from desispec.workflow.submission import submit_linkcal_jobs


def _make_refnight_ptable(refnight, jobdescs_qids):
    """Create a minimal reference-night processing table.

    Args:
        refnight (int): Night integer in YYYYMMDD format.
        jobdescs_qids (list of (str, int)): List of (JOBDESC, LATEST_QID) pairs.
            Set LATEST_QID > get_default_qid() to simulate an in-flight job.

    Returns:
        Table: Processing table with one row per (JOBDESC, LATEST_QID) pair.
    """
    ptable = instantiate_processing_table()
    default_qid = get_default_qid()
    for intid, (jobdesc, qid) in enumerate(jobdescs_qids):
        prow = default_prow()
        prow['INTID'] = intid
        prow['JOBDESC'] = jobdesc
        prow['NIGHT'] = refnight
        prow['LATEST_QID'] = qid
        prow['STATUS'] = 'COMPLETED' if qid == default_qid else 'RUNNING'
        prow['EXPID'] = np.array([intid * 100 + 1], dtype=int)
        prow['PROCCAMWORD'] = 'a0123456789'
        ptable.add_row(prow)
    return ptable


class TestLinkcalCrossNightDependencies(unittest.TestCase):
    """Regression tests for cross-night dependency logic in submit_linkcal_jobs."""

    @classmethod
    def setUpClass(cls):
        cls.refnight = 20230913
        cls.night = 20230914
        cls.reduxdir = tempfile.mkdtemp()
        cls.specprod = 'test'
        cls.proddir = os.path.join(cls.reduxdir, cls.specprod)
        os.makedirs(cls.proddir)

        cls.origenv = os.environ.copy()
        os.environ['DESI_SPECTRO_REDUX'] = cls.reduxdir
        os.environ['SPECPROD'] = cls.specprod
        os.environ['NERSC_HOST'] = 'perlmutter'

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.reduxdir)
        for key in ('DESI_SPECTRO_REDUX', 'SPECPROD', 'NERSC_HOST'):
            if key in cls.origenv:
                os.environ[key] = cls.origenv[key]
            elif key in os.environ:
                del os.environ[key]

    def setUp(self):
        self.refptable_path = findfile('proctable', night=self.refnight)
        os.makedirs(os.path.dirname(self.refptable_path), exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.refptable_path):
            os.remove(self.refptable_path)

    def _write_refnight_ptable(self, jobdescs_qids):
        """Build and write the reference-night proctable to disk.

        Args:
            jobdescs_qids (list of (str, int)): Passed to _make_refnight_ptable.

        Returns:
            Table: The written proctable (useful for INTID assertions).
        """
        ptable = _make_refnight_ptable(self.refnight, jobdescs_qids)
        write_table(ptable, tablename=self.refptable_path, tabletype='proctable')
        return ptable

    def _run_linkcal(self, include, refnight=None):
        """Call submit_linkcal_jobs with minimal setup.

        Args:
            include (str): Comma-separated calibration file types to include.
            refnight (int or None): Reference night; defaults to self.refnight.

        Returns:
            Table: The returned processing table containing the linkcal row.
        """
        if refnight is None:
            refnight = self.refnight
        cal_override = {'linkcal': {'refnight': refnight, 'include': include}}
        ptable = instantiate_processing_table()
        ptable, _ = submit_linkcal_jobs(
            self.night, ptable,
            cal_override=cal_override,
            dry_run_level=4,
            check_outputs=False,
        )
        return ptable

    # ------------------------------------------------------------------
    # Helpers to extract the linkcal row from the returned ptable
    # ------------------------------------------------------------------

    def _get_linkcal_row(self, ptable):
        linkcal_mask = ptable['JOBDESC'] == 'linkcal'
        self.assertEqual(np.sum(linkcal_mask), 1,
                         "Expected exactly one linkcal row in ptable")
        return ptable[linkcal_mask][0]

    # ==================================================================
    # Tests
    # ==================================================================

    def test_no_dependency_when_refnight_ptable_missing(self):
        """No cross-night dependency is added when the refnight proctable is absent."""
        # Do not write a refnight proctable
        self.assertFalse(os.path.exists(self.refptable_path))
        ptable = self._run_linkcal('biasnight')
        linkcal = self._get_linkcal_row(ptable)
        self.assertEqual(len(linkcal['INT_DEP_IDS']), 0)
        self.assertEqual(len(linkcal['LATEST_DEP_QID']), 0)

    def test_single_filetype_inflight_dependency(self):
        """Linkcal prow picks up a single in-flight dependency from refnight.

        When 'biasnight' is being linked and the refnight proctable contains a
        running biasnight job, the new-night linkcal row must list that job's
        INTID in INT_DEP_IDS and its LATEST_QID in LATEST_DEP_QID.
        """
        bias_qid = 12345
        refptab = self._write_refnight_ptable([('biasnight', bias_qid)])
        bias_intid = refptab[refptab['JOBDESC'] == 'biasnight']['INTID'][0]

        ptable = self._run_linkcal('biasnight')
        linkcal = self._get_linkcal_row(ptable)

        self.assertIn(bias_intid, linkcal['INT_DEP_IDS'])
        self.assertIn(bias_qid, linkcal['LATEST_DEP_QID'])

    def test_single_filetype_completed_no_qid_dependency(self):
        """A COMPLETED refnight job appears in INT_DEP_IDS but not LATEST_DEP_QID.

        A job that has already completed is still tracked (INT_DEP_IDS) but
        should not block the linkcal submission (LATEST_DEP_QID excluded).
        """
        default_qid = get_default_qid()
        refptab = self._write_refnight_ptable([('biasnight', default_qid)])
        bias_intid = refptab[refptab['JOBDESC'] == 'biasnight']['INTID'][0]

        ptable = self._run_linkcal('biasnight')
        linkcal = self._get_linkcal_row(ptable)

        self.assertIn(bias_intid, linkcal['INT_DEP_IDS'])
        # COMPLETED job must not appear in LATEST_DEP_QID
        self.assertEqual(len(linkcal['LATEST_DEP_QID']), 0)

    def test_multiple_filetypes_multiple_dependencies(self):
        """Linkcal prow gets one dependency per linked filetype.

        When both 'biasnight' and 'fiberflatnight' are linked and each has a
        corresponding in-flight job in the refnight proctable, both must
        appear in INT_DEP_IDS and LATEST_DEP_QID.
        """
        bias_qid = 22222
        flat_qid = 33333
        refptab = self._write_refnight_ptable([
            ('biasnight', bias_qid),
            ('nightlyflat', flat_qid),
        ])
        bias_intid = refptab[refptab['JOBDESC'] == 'biasnight']['INTID'][0]
        flat_intid = refptab[refptab['JOBDESC'] == 'nightlyflat']['INTID'][0]

        ptable = self._run_linkcal('biasnight,fiberflatnight')

        linkcal = self._get_linkcal_row(ptable)

        self.assertIn(bias_intid, linkcal['INT_DEP_IDS'])
        self.assertIn(flat_intid, linkcal['INT_DEP_IDS'])
        self.assertIn(bias_qid, linkcal['LATEST_DEP_QID'])
        self.assertIn(flat_qid, linkcal['LATEST_DEP_QID'])

    def test_refnight_linkcal_job_added_as_dependency(self):
        """A linkcal job in the refnight proctable is itself added as a dependency.

        If the refnight had its own linkcal job (chained calibrations), it must
        be included in the new-night linkcal's INT_DEP_IDS / LATEST_DEP_QID.
        """
        linkcal_qid = 44444
        bias_qid = 55555
        refptab = self._write_refnight_ptable([
            ('biasnight', bias_qid),
            ('linkcal', linkcal_qid),
        ])
        linkcal_intid = refptab[refptab['JOBDESC'] == 'linkcal']['INTID'][0]
        bias_intid = refptab[refptab['JOBDESC'] == 'biasnight']['INTID'][0]

        ptable = self._run_linkcal('biasnight')

        newnight_linkcal = self._get_linkcal_row(ptable)

        self.assertIn(bias_intid, newnight_linkcal['INT_DEP_IDS'])
        self.assertIn(linkcal_intid, newnight_linkcal['INT_DEP_IDS'])
        self.assertIn(bias_qid, newnight_linkcal['LATEST_DEP_QID'])
        self.assertIn(linkcal_qid, newnight_linkcal['LATEST_DEP_QID'])

    def test_no_dependency_when_refnight_job_absent_from_ptable(self):
        """No dependency is added when the linked filetype has no matching job.

        If the refnight proctable does not contain the job corresponding to the
        linked filetype, the linkcal row must have no dependencies.
        """
        # Refnight only has a tilenight job, not a biasnight job
        refptab = self._write_refnight_ptable([('tilenight', 99999)])

        ptable = self._run_linkcal('biasnight')
        linkcal = self._get_linkcal_row(ptable)

        self.assertEqual(len(linkcal['INT_DEP_IDS']), 0)
        self.assertEqual(len(linkcal['LATEST_DEP_QID']), 0)


if __name__ == '__main__':
    unittest.main()
