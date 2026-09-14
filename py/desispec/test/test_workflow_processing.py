"""
Test desispec.workflow.processing
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from desispec.workflow.processing import submit_batch_script, \
    check_calibration_bundle_steps_on_disk, create_batch_script, \
    desi_proc_command, desi_proc_joint_fit_command, get_jobdesc_to_file_map, \
    make_calibration_bundle_prow, make_joint_prow
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


def make_erow(expid, night=20250318, obstype='arc', camword='a0123456789',
              badcamword='', badamps='', program='calib short arcs all'):
    """Create a minimal exposure table row dict for bundle construction tests"""
    return {'EXPID': expid, 'NIGHT': night, 'OBSTYPE': obstype,
            'CAMWORD': camword, 'BADCAMWORD': badcamword, 'BADAMPS': badamps,
            'TILEID': -99, 'LASTSTEP': 'all', 'PROGRAM': program}


def make_calibjobs(**kwargs):
    """Create a calibjobs dict where every entry is None unless overridden"""
    calibjobs = {'biasnight': None, 'biaspdark': None, 'ccdcalib': None,
                 'psfnight': None, 'nightlyflat': None, 'linkcal': None}
    calibjobs.update(kwargs)
    return calibjobs


def make_calibration_job(intid, jobdesc):
    """Create a stand-in processing row for an upstream calibration job"""
    prow = default_prow()
    prow['INTID'] = intid
    prow['JOBDESC'] = jobdesc
    prow['LATEST_QID'] = 1000 + intid
    prow['STATUS'] = 'SUBMITTED'
    return prow


class TestCalibrationBundleRows(unittest.TestCase):
    """Tests for the processing table representation of calibration bundles"""

    def test_make_calibration_bundle_prow_arcs(self):
        """Five arcs give one psfnight row holding all five EXPIDs"""
        ccdcalib = make_calibration_job(100, 'ccdcalib')
        calibjobs = make_calibjobs(ccdcalib=ccdcalib)
        erows = [make_erow(expid) for expid in [104, 100, 103, 101, 102]]

        bundle, steps, next_id = make_calibration_bundle_prow(
                erows, descriptor='psfnight', internal_id=7, calibjobs=calibjobs)

        self.assertEqual(bundle['JOBDESC'], 'psfnight')
        self.assertEqual(bundle['OBSTYPE'], 'arc')
        self.assertEqual(bundle['CALIBRATOR'], 1)
        self.assertEqual(bundle['INTID'], 7)
        self.assertEqual(next_id, 8, 'a bundle consumes exactly one INTID')
        self.assertEqual(sorted(bundle['EXPID']), [100, 101, 102, 103, 104])
        self.assertEqual(bundle['PROCCAMWORD'], 'a0123456789')

        ## the bundle depends on the real upstream job, not the temporary rows
        self.assertEqual(list(bundle['INT_DEP_IDS']), [ccdcalib['INTID']])

        ## the steps are ordered by EXPID and are ordinary arc jobs
        self.assertEqual([step['EXPID'][0] for step in steps],
                         [100, 101, 102, 103, 104])
        for step in steps:
            self.assertEqual(step['JOBDESC'], 'arc')

    def test_make_calibration_bundle_prow_flats(self):
        """A nightlyflat bundle uses the intersection of the flat camwords"""
        psfnight = make_calibration_job(200, 'psfnight')
        calibjobs = make_calibjobs(psfnight=psfnight)
        erows = [make_erow(300, obstype='flat', camword='a0123456789'),
                 make_erow(301, obstype='flat', camword='a012345678'),
                 make_erow(302, obstype='flat', camword='a0123456789')]

        bundle, steps, next_id = make_calibration_bundle_prow(
                erows, descriptor='nightlyflat', internal_id=3,
                calibjobs=calibjobs)

        self.assertEqual(bundle['JOBDESC'], 'nightlyflat')
        self.assertEqual(bundle['OBSTYPE'], 'flat')
        self.assertEqual(sorted(bundle['EXPID']), [300, 301, 302])
        self.assertEqual(bundle['PROCCAMWORD'], 'a012345678')
        self.assertEqual(list(bundle['INT_DEP_IDS']), [psfnight['INTID']])
        for step in steps:
            self.assertEqual(step['JOBDESC'], 'flat')

    def test_make_calibration_bundle_prow_cteflats(self):
        """A cteflat bundle uses the union of its exposures' camwords"""
        psfnight = make_calibration_job(200, 'psfnight')
        nightlyflat = make_calibration_job(201, 'nightlyflat')
        calibjobs = make_calibjobs(psfnight=psfnight, nightlyflat=nightlyflat)
        erows = [make_erow(400, obstype='flat', camword='a012345678',
                           program='led03 10s flat for cte check'),
                 make_erow(401, obstype='flat', camword='a0123456789',
                           program='led03 3s flat for cte check'),
                 make_erow(402, obstype='flat', camword='a012345678',
                           program='led03 1s flat for cte check')]

        bundle, steps, next_id = make_calibration_bundle_prow(
                erows, descriptor='cteflat', internal_id=11,
                calibjobs=calibjobs)

        self.assertEqual(bundle['JOBDESC'], 'cteflat')
        self.assertEqual(bundle['OBSTYPE'], 'flat')
        self.assertEqual(bundle['CALIBRATOR'], 1)
        self.assertEqual(sorted(bundle['EXPID']), [400, 401, 402])
        ## union, not intersection: no joint fit imposes a common camera set
        self.assertEqual(bundle['PROCCAMWORD'], 'a0123456789')
        ## CTE flats need a PSF, but not fiberflatnight
        self.assertEqual(list(bundle['INT_DEP_IDS']), [psfnight['INTID']])
        ## the steps keep their own camwords and are ordinary flat jobs
        self.assertEqual([step['PROCCAMWORD'] for step in steps],
                         ['a012345678', 'a0123456789', 'a012345678'])
        for step in steps:
            self.assertEqual(step['JOBDESC'], 'flat')

    def test_make_calibration_bundle_prow_errors(self):
        """Unknown descriptors and empty exposure lists are rejected"""
        calibjobs = make_calibjobs()
        with self.assertRaises(ValueError):
            make_calibration_bundle_prow([make_erow(100)], descriptor='flat',
                                         internal_id=1, calibjobs=calibjobs)
        with self.assertRaises(ValueError):
            make_calibration_bundle_prow([], descriptor='psfnight',
                                         internal_id=1, calibjobs=calibjobs)

    def test_make_joint_prow_cteflat(self):
        """make_joint_prow understands cteflat without warning"""
        prows = []
        for expid, camword in ((400, 'a012345678'), (401, 'a0123456789')):
            prow = default_prow()
            prow['EXPID'] = np.array([expid])
            prow['PROCCAMWORD'] = camword
            prow['OBSTYPE'] = 'flat'
            prow['JOBDESC'] = 'flat'
            prows.append(prow)

        with patch('desispec.workflow.processing.get_logger') as mock_logger:
            joint, next_id = make_joint_prow(prows, descriptor='cteflat',
                                             internal_id=5)
            mock_logger.return_value.warning.assert_not_called()

        self.assertEqual(joint['JOBDESC'], 'cteflat')
        self.assertEqual(sorted(joint['EXPID']), [400, 401])
        self.assertEqual(joint['PROCCAMWORD'], 'a0123456789')

    def test_cteflat_completion_product(self):
        """cteflat is completed by frames, not by fiberflats"""
        job_to_file_map = get_jobdesc_to_file_map()
        self.assertEqual(job_to_file_map['cteflat'], 'frame')
        self.assertEqual(job_to_file_map['flat'], 'fiberflat')


class TestBundleCommands(unittest.TestCase):
    """Tests for the direct execution mode of the command builders"""

    def _make_prow(self, jobdesc='flat', obstype='flat', expids=(100,),
                   camword='a0123456789', badamps=''):
        prow = default_prow()
        prow['NIGHT'] = 20250318
        prow['EXPID'] = np.array(expids)
        prow['OBSTYPE'] = obstype
        prow['JOBDESC'] = jobdesc
        prow['PROCCAMWORD'] = camword
        prow['BADAMPS'] = badamps
        return prow

    def test_desi_proc_command_default_unchanged(self):
        """The default mode still asks desi_proc to write another batch script"""
        prow = self._make_prow(jobdesc='arc', obstype='arc')
        cmd = desi_proc_command(prow, system_name='perlmutter-cpu')
        self.assertIn('--batch', cmd)
        self.assertIn('--nosubmit', cmd)
        self.assertNotIn('--mpi', cmd)
        self.assertIn('--cameras=a0123456789', cmd)

    def test_desi_proc_command_direct_mode(self):
        """Direct mode drops the batch options and runs under MPI"""
        prow = self._make_prow(jobdesc='arc', obstype='arc', badamps='b7D')
        cmd = desi_proc_command(prow, system_name='perlmutter-cpu',
                                queue='realtime', direct_mode=True)
        self.assertNotIn('--batch', cmd)
        self.assertNotIn('--nosubmit', cmd)
        self.assertNotIn('-q ', cmd)
        self.assertIn(' --mpi', cmd)
        self.assertIn('--cameras a0123456789', cmd)
        self.assertIn('-n 20250318', cmd)
        self.assertIn('-e 100', cmd)
        self.assertIn('--badamps=b7D', cmd)

    def test_desi_proc_command_direct_mode_use_specter(self):
        """use-specter is preserved for flat steps in direct mode"""
        prow = self._make_prow(jobdesc='flat', obstype='flat')
        cmd = desi_proc_command(prow, system_name='perlmutter-gpu',
                                use_specter=True, direct_mode=True)
        self.assertIn('--use-specter', cmd)

    def test_desi_proc_joint_fit_command_direct_mode(self):
        """The joint fit command runs under MPI and lists every EXPID"""
        prow = self._make_prow(jobdesc='nightlyflat', obstype='flat',
                               expids=(100, 101, 102))
        cmd = desi_proc_joint_fit_command(prow, queue='realtime',
                                          direct_mode=True)
        self.assertNotIn('--batch', cmd)
        self.assertNotIn('--nosubmit', cmd)
        self.assertIn(' --mpi', cmd)
        self.assertIn('--obstype flat', cmd)
        self.assertIn('--cameras a0123456789', cmd)
        self.assertIn('-e 100,101,102', cmd)

    def test_create_batch_script_bundle_guard(self):
        """The bundle branch only fires when the bundle metadata is provided"""
        prow = self._make_prow(jobdesc='nightlyflat', obstype='flat',
                               expids=(100, 101))
        step = self._make_prow(jobdesc='flat', obstype='flat', expids=(100,))

        ## without the reserved key, an ordinary joint-fit script is created
        with patch('desispec.workflow.processing.batch_script_pathname',
                   return_value='/fake/scripts/nightlyflat.slurm'), \
             patch('desispec.workflow.processing.create_calibration_bundle_batch_script') as mock_bundle:
            create_batch_script(prow, dry_run=2, joint=True,
                                system_name='perlmutter-gpu')
            mock_bundle.assert_not_called()

        ## with it, the bundle writer is used
        extra_job_args = {'calibration_bundle_steps': [step],
                          'bundle_full_camword': 'a0123456789'}
        with patch('desispec.workflow.processing.check_calibration_bundle_steps_on_disk',
                   return_value=[step]), \
             patch('desispec.workflow.processing.create_calibration_bundle_batch_script',
                   return_value='/fake/scripts/nightlyflat.slurm') as mock_bundle:
            create_batch_script(prow, dry_run=0, joint=True,
                                system_name='perlmutter-gpu',
                                extra_job_args=extra_job_args)
            mock_bundle.assert_called_once()
            kwargs = mock_bundle.call_args[1]
            self.assertEqual(kwargs['jobdesc'], 'nightlyflat')
            self.assertEqual(kwargs['camword'], 'a0123456789')
            self.assertEqual(len(kwargs['steps']), 1)
            self.assertIn('desi_proc_joint_fit', kwargs['joint_cmd'])
            self.assertIn(' --mpi', kwargs['joint_cmd'])


class TestCalibrationBundleOutputChecks(unittest.TestCase):
    """Tests for pruning bundle steps whose products already exist"""

    @classmethod
    def setUpClass(cls):
        cls.origenv = os.environ.copy()
        cls.reduxdir = tempfile.mkdtemp()
        cls.specprod = 'test'
        os.environ['DESI_SPECTRO_REDUX'] = cls.reduxdir
        os.environ['SPECPROD'] = cls.specprod
        os.environ['NERSC_HOST'] = 'perlmutter'
        os.makedirs(os.path.join(cls.reduxdir, cls.specprod))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.reduxdir)
        for key in ('DESI_SPECTRO_REDUX', 'SPECPROD', 'NERSC_HOST'):
            if key in cls.origenv:
                os.environ[key] = cls.origenv[key]
            elif key in os.environ:
                del os.environ[key]

    def setUp(self):
        self.night = 20250318
        self.calibjobs = make_calibjobs()

    def tearDown(self):
        proddir = os.path.join(self.reduxdir, self.specprod)
        for name in os.listdir(proddir):
            path = os.path.join(proddir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    def _touch(self, filetype, expid, camera):
        from desispec.io import findfile
        pathname = findfile(filetype, night=self.night, expid=expid,
                            camera=camera)
        os.makedirs(os.path.dirname(pathname), exist_ok=True)
        open(pathname, 'w').close()

    def _cte_bundle(self, camwords):
        erows = [make_erow(400 + i, night=self.night, obstype='flat',
                           camword=camword,
                           program='led03 flat for cte check')
                 for i, camword in enumerate(camwords)]
        return make_calibration_bundle_prow(erows, descriptor='cteflat',
                                            internal_id=1,
                                            calibjobs=self.calibjobs)

    def test_cteflat_steps_use_their_own_camwords(self):
        """Each CTE step keeps its own camword rather than the bundle union"""
        bundle, steps, _ = self._cte_bundle(['a01', 'a12', 'a2'])
        self.assertEqual(bundle['PROCCAMWORD'], 'a012')

        remaining = check_calibration_bundle_steps_on_disk(bundle, steps)
        self.assertEqual([step['PROCCAMWORD'] for step in remaining],
                         ['a01', 'a12', 'a2'])

    def test_cteflat_steps_pruned_by_existing_frames(self):
        """Existing frames remove only the matching exposure and cameras"""
        bundle, steps, _ = self._cte_bundle(['a01', 'a12', 'a2'])

        ## exposure 400 is fully done, exposure 401 is half done
        for camera in ('b0', 'r0', 'z0', 'b1', 'r1', 'z1'):
            self._touch('frame', 400, camera)
        for camera in ('b1', 'r1', 'z1'):
            self._touch('frame', 401, camera)

        remaining = check_calibration_bundle_steps_on_disk(bundle, steps)
        self.assertEqual([step['EXPID'][0] for step in remaining], [401, 402])
        self.assertEqual([step['PROCCAMWORD'] for step in remaining],
                         ['a2', 'a2'])

    def test_cteflat_preproc_is_not_enough(self):
        """preproc files alone do not complete a CTE step; frames are required"""
        bundle, steps, _ = self._cte_bundle(['a0'])
        ## neither the preproc that night QA reads nor ccdcalib's ctepreproc
        ## count as a completed CTE step
        for camera in ('b0', 'r0', 'z0'):
            self._touch('preproc', 400, camera)
            self._touch('preproc_for_cte', 400, camera)

        remaining = check_calibration_bundle_steps_on_disk(bundle, steps)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]['PROCCAMWORD'], 'a0')

        for camera in ('b0', 'r0', 'z0'):
            self._touch('frame', 400, camera)
        self.assertEqual(len(check_calibration_bundle_steps_on_disk(bundle, steps)), 0)

    def test_arc_steps_intersected_with_bundle_camword(self):
        """Arc steps don't process cameras that can't contribute to psfnight"""
        erows = [make_erow(100 + i, night=self.night, camword=camword)
                 for i, camword in enumerate(['a0123456789', 'a0123456789',
                                              'a0123456789', 'a012345678',
                                              'a012345678'])]
        bundle, steps, _ = make_calibration_bundle_prow(
                erows, descriptor='psfnight', internal_id=1,
                calibjobs=self.calibjobs)
        ## camera 9 is in 3 of the 5 arcs so psfnight still fits it
        self.assertEqual(bundle['PROCCAMWORD'], 'a0123456789')

        ## if psfnight is pruned to a0, so is every arc step
        bundle['PROCCAMWORD'] = 'a0'
        remaining = check_calibration_bundle_steps_on_disk(bundle, steps)
        self.assertEqual(len(remaining), 5)
        for step in remaining:
            self.assertEqual(step['PROCCAMWORD'], 'a0')

    def test_arc_steps_dropped_when_fitpsf_exists(self):
        """A step with all of its fitpsf files is left out of the script"""
        erows = [make_erow(100 + i, night=self.night, camword='a0')
                 for i in range(3)]
        bundle, steps, _ = make_calibration_bundle_prow(
                erows, descriptor='psfnight', internal_id=1,
                calibjobs=self.calibjobs)
        for camera in ('b0', 'r0', 'z0'):
            self._touch('fitpsf', 100, camera)

        remaining = check_calibration_bundle_steps_on_disk(bundle, steps)
        self.assertEqual([step['EXPID'][0] for step in remaining], [101, 102])


if __name__ == '__main__':
    unittest.main()
