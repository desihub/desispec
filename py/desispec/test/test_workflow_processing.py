"""
Test desispec.workflow.processing
"""

import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from desispec.workflow.processing import submit_batch_script
from desispec.workflow.proctable import default_prow, get_err_qid, get_default_qid
from desispec.workflow.queue import clear_queue_state_cache


class TestSubmitBatchScript(unittest.TestCase):
    """Tests for submit_batch_script focusing on dependency failure and sbatch
    failure handling (issue #2656 scenario).
    """

    def setUp(self):
        clear_queue_state_cache()

    def _make_prow(self, jobdesc='flat', dep_qids=None):
        """Create a minimal processing row for testing.

        Args:
            jobdesc (str): Job description (JOBDESC column value).
            dep_qids (array-like or None): Dependency QIDs (LATEST_DEP_QID).

        Returns:
            dict: A default processing row with the given fields set.
        """
        prow = default_prow()
        prow['JOBDESC'] = jobdesc
        prow['NIGHT'] = 20211102
        if dep_qids is not None:
            prow['LATEST_DEP_QID'] = np.array(dep_qids, dtype=int)
        return prow

    def test_submit_batch_script_err_qid_dependency(self):
        """A dependency with err_qid should block submission and set UNSUBMITTED.

        Regression test for issue #2656: when an upstream job was itself
        unsubmitted (recorded with LATEST_QID=get_err_qid()), downstream jobs
        that depend on it must also be blocked.
        """
        err_qid = get_err_qid()
        prow = self._make_prow(dep_qids=[err_qid])

        with patch('desispec.workflow.processing.batch_script_pathname',
                   return_value='/fake/scripts/flat.slurm'):
            result = submit_batch_script(prow, dry_run=4)

        self.assertEqual(result['STATUS'], 'UNSUBMITTED')
        self.assertEqual(result['LATEST_QID'], err_qid)
        self.assertEqual(len(result['ALL_QIDS']), 0)

    def test_submit_batch_script_failed_state_dependency(self):
        """A dependency in a bad final state (e.g. FAILED) should block submission.

        Regression test for issue #2656: downstream jobs must not be submitted
        when an upstream job is in a failed terminal state.
        """
        err_qid = get_err_qid()
        dep_qid = 12345
        prow = self._make_prow(dep_qids=[dep_qid])

        with patch('desispec.workflow.processing.get_queue_states_from_qids',
                   return_value={dep_qid: 'FAILED'}):
            with patch('desispec.workflow.processing.batch_script_pathname',
                       return_value='/fake/scripts/flat.slurm'):
                result = submit_batch_script(prow, dry_run=4)

        self.assertEqual(result['STATUS'], 'UNSUBMITTED')
        self.assertEqual(result['LATEST_QID'], err_qid)
        self.assertEqual(len(result['ALL_QIDS']), 0)

    def test_submit_batch_script_sbatch_failure(self):
        """When sbatch fails repeatedly the job should be recorded as UNSUBMITTED.

        Regression test for issue #2656: if sbatch itself fails after the
        maximum number of retries, the job must be marked UNSUBMITTED with
        LATEST_QID=get_err_qid() so that downstream jobs are also blocked.
        """
        err_qid = get_err_qid()
        prow = self._make_prow()

        sbatch_error = subprocess.CalledProcessError(
            returncode=1, cmd=['sbatch'], output='sbatch: error')

        with patch('desispec.workflow.processing.batch_script_pathname',
                   return_value='/fake/scripts/flat.slurm'):
            with patch('subprocess.check_output', side_effect=sbatch_error):
                with patch('time.sleep'):  # avoid real 60s delays between retries
                    result = submit_batch_script(prow, dry_run=0)

        self.assertEqual(result['STATUS'], 'UNSUBMITTED')
        self.assertEqual(result['LATEST_QID'], err_qid)
        self.assertEqual(len(result['ALL_QIDS']), 0)

    def test_submit_batch_script_dry_run_succeeds(self):
        """In dry-run mode with no failing dependencies, submission should succeed."""
        default_qid = get_default_qid()
        prow = self._make_prow()

        with patch('desispec.workflow.processing.batch_script_pathname',
                   return_value='/fake/scripts/flat.slurm'):
            result = submit_batch_script(prow, dry_run=1)

        self.assertEqual(result['STATUS'], 'SUBMITTED')
        self.assertNotEqual(result['LATEST_QID'], get_err_qid())
        self.assertNotEqual(result['LATEST_QID'], default_qid)
        self.assertEqual(len(result['ALL_QIDS']), 1)


if __name__ == '__main__':
    unittest.main()
