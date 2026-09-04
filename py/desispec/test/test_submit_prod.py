"""
Test reference-night discovery in desispec.scripts.submit_prod.

An override file normally links calibrations from an *earlier* night, which the
chronological submission order in submit_production() handles for free.
Occasionally an override points forward in time, e.g. when a night's flats are
bad but the following night's are good. Those reference nights have to be
calibrated ahead of the normal order, which is what these functions identify.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from astropy.table import Table

from contextlib import contextmanager

from desispec.io.meta import findfile
from desispec.workflow.proctable import get_default_qid, get_err_qid
from desispec.workflow.queue import get_resubmission_states
from desispec.scripts.submit_prod import (
    bias_dependency_available,
    get_linkcal_refnight,
    get_refnights_needing_early_calibration,
    submit_early_refnight_calibrations,
    submit_production,
)


@contextmanager
def _null_redirect(*args, **kwargs):
    """
    Stand-in for stdouterr_redirected().

    The real one redirects at the file descriptor level, which collides with
    pytest's output capture. The orchestration logic under test does not depend
    on it.
    """
    yield


class TestEarlyRefnightDiscovery(unittest.TestCase):
    """Tests for get_linkcal_refnight and get_refnights_needing_early_calibration."""

    @classmethod
    def setUpClass(cls):
        cls.reduxdir = tempfile.mkdtemp()
        cls.specprod = 'test'
        cls.proddir = os.path.join(cls.reduxdir, cls.specprod)
        os.makedirs(cls.proddir)

        cls.origenv = os.environ.copy()
        os.environ['DESI_SPECTRO_REDUX'] = cls.reduxdir
        os.environ['SPECPROD'] = cls.specprod

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.reduxdir)
        for key in ('DESI_SPECTRO_REDUX', 'SPECPROD'):
            if key in cls.origenv:
                os.environ[key] = cls.origenv[key]
            elif key in os.environ:
                del os.environ[key]

    def tearDown(self):
        exptabdir = os.path.join(self.proddir, 'exposure_tables')
        if os.path.exists(exptabdir):
            shutil.rmtree(exptabdir)

    def _write_override(self, night, refnight, include=None, exclude=None):
        """Write an override file linking calibrations from refnight"""
        linkcal = {'refnight': refnight}
        if include is not None:
            linkcal['include'] = include
        if exclude is not None:
            linkcal['exclude'] = exclude
        pathname = findfile('override', night=night)
        os.makedirs(os.path.dirname(pathname), exist_ok=True)
        with open(pathname, 'w') as fil:
            fil.write('calibration:\n    linkcal:\n')
            for key, val in linkcal.items():
                fil.write(f'        {key}: {val}\n')

    def _write_exptable(self, night):
        """Touch an exposure table so the night looks processable"""
        pathname = findfile('exposure_table', night=night)
        os.makedirs(os.path.dirname(pathname), exist_ok=True)
        open(pathname, 'w').close()

    # ==================================================================
    # get_linkcal_refnight
    # ==================================================================

    def test_no_override_file(self):
        """A night without an override file links nothing"""
        self.assertEqual(get_linkcal_refnight(20230914), (None, set()))

    def test_override_without_linkcal(self):
        """An override file that doesn't link calibrations returns None"""
        pathname = findfile('override', night=20230914)
        os.makedirs(os.path.dirname(pathname), exist_ok=True)
        with open(pathname, 'w') as fil:
            fil.write('calibration:\n  nightlyflat:\n'
                      + '    extra_cmd_args: [--autocal-ff-solve-grad]\n')
        self.assertEqual(get_linkcal_refnight(20230914), (None, set()))

    def test_malformed_include_raises(self):
        """A yaml list include is not supported and must not pass silently

        derive_include_exclude() expects a comma separated string, so a yaml
        list would fail later at submission time. Since every night's override
        file is inspected up front, it needs to fail here rather than be
        quietly treated as linking nothing.
        """
        pathname = findfile('override', night=20230914)
        os.makedirs(os.path.dirname(pathname), exist_ok=True)
        with open(pathname, 'w') as fil:
            fil.write('calibration:\n    linkcal:\n        refnight: 20230915\n'
                      + '        include: [biasnight, badcolumns]\n')
        with self.assertRaises(AttributeError):
            get_linkcal_refnight(20230914)

    def test_include_and_exclude_together_raises(self):
        """include and exclude are mutually exclusive"""
        self._write_override(20230914, 20230915, include='fiberflatnight',
                             exclude='biasnight')
        with self.assertRaises(ValueError):
            get_linkcal_refnight(20230914)

    def test_include_is_resolved(self):
        """An explicit include list is returned as the set of linked files"""
        self._write_override(20230914, 20230913, include='biasnight')
        refnight, files_to_link = get_linkcal_refnight(20230914)
        self.assertEqual(refnight, 20230913)
        self.assertEqual(files_to_link, {'biasnight'})

    def test_exclude_is_resolved(self):
        """An exclude list resolves to everything else, including biasnight"""
        self._write_override(20230914, 20230913, exclude='fiberflatnight')
        refnight, files_to_link = get_linkcal_refnight(20230914)
        self.assertEqual(refnight, 20230913)
        self.assertIn('biasnight', files_to_link)
        self.assertNotIn('fiberflatnight', files_to_link)

    # ==================================================================
    # get_refnights_needing_early_calibration
    # ==================================================================

    def test_earlier_refnight_needs_nothing(self):
        """Linking from an earlier night is handled by chronological order"""
        self._write_exptable(20230913)
        self._write_exptable(20230914)
        self._write_override(20230914, 20230913, include='biasnight')
        self.assertEqual(
            get_refnights_needing_early_calibration([20230913, 20230914]), [])

    def test_later_refnight_with_biasnight(self):
        """Linking biasnight forward requires the bias be submitted first"""
        self._write_exptable(20230914)
        self._write_exptable(20230915)
        self._write_override(20230914, 20230915, include='biasnight')
        self.assertEqual(
            get_refnights_needing_early_calibration([20230914, 20230915]),
            [(20230915, True)])

    def test_later_refnight_without_biasnight(self):
        """Linking something other than biasnight doesn't need a bias-only pass"""
        self._write_exptable(20230914)
        self._write_exptable(20230915)
        self._write_override(20230914, 20230915, include='fiberflatnight')
        self.assertEqual(
            get_refnights_needing_early_calibration([20230914, 20230915]),
            [(20230915, False)])

    def test_refnight_without_exposure_table_is_skipped(self):
        """A refnight with no exposure table can't be processed, so skip it"""
        self._write_exptable(20230914)
        self._write_override(20230914, 20230915, include='biasnight')
        self.assertEqual(
            get_refnights_needing_early_calibration([20230914]), [])

    def test_chain_puts_dependencies_first(self):
        """A refnight that itself links from another night comes after it"""
        for night in (20230910, 20230914, 20230915):
            self._write_exptable(night)
        ## 20230914 links forward to 20230915, which itself links back to 20230910
        self._write_override(20230914, 20230915, include='biasnight')
        self._write_override(20230915, 20230910, include='fiberflatnight')
        self.assertEqual(
            get_refnights_needing_early_calibration([20230914, 20230915]),
            [(20230910, False), (20230915, True)])

    def test_circular_reference_raises(self):
        """A circular set of overrides aborts rather than inventing an order

        A cycle cannot be topologically ordered, so there is no correct
        submission order. Proceeding would submit one side before the
        calibrations it links from, which is the failure the pre-pass exists to
        prevent, so it must abort before anything is submitted.
        """
        self._write_exptable(20230914)
        self._write_exptable(20230915)
        self._write_override(20230914, 20230915, include='fiberflatnight')
        self._write_override(20230915, 20230914, include='fiberflatnight')
        with self.assertRaises(ValueError):
            get_refnights_needing_early_calibration([20230914, 20230915])

    def test_long_chain_is_not_mistaken_for_a_cycle(self):
        """A three-deep acyclic chain still resolves, dependencies first"""
        for night in (20230910, 20230912, 20230914, 20230915):
            self._write_exptable(night)
        ## 20230914 -> 20230915 -> 20230912 -> 20230910, no cycle
        self._write_override(20230914, 20230915, include='biasnight')
        self._write_override(20230915, 20230912, include='fiberflatnight')
        self._write_override(20230912, 20230910, include='fiberflatnight')
        self.assertEqual(
            get_refnights_needing_early_calibration([20230914, 20230915]),
            [(20230910, False), (20230912, False), (20230915, True)])

    def test_multiple_nights_share_one_refnight(self):
        """Two nights linking from the same refnight yield a single entry"""
        for night in (20230913, 20230914, 20230915):
            self._write_exptable(night)
        self._write_override(20230913, 20230915, include='fiberflatnight')
        self._write_override(20230914, 20230915, include='biasnight')
        self.assertEqual(
            get_refnights_needing_early_calibration([20230913, 20230914]),
            [(20230915, True)])


class TestBiasDependencyAvailable(unittest.TestCase):
    """
    Tests for bias_dependency_available().

    An earlier night linking 'biasnight' from a reference night needs a job on
    that reference night which actually produces or provides the biasnight. A
    row merely existing is not enough.
    """

    @classmethod
    def setUpClass(cls):
        cls.reduxdir = tempfile.mkdtemp()
        cls.specprod = 'test'
        cls.proddir = os.path.join(cls.reduxdir, cls.specprod)
        os.makedirs(cls.proddir)
        cls.origenv = os.environ.copy()
        os.environ['DESI_SPECTRO_REDUX'] = cls.reduxdir
        os.environ['SPECPROD'] = cls.specprod

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.reduxdir)
        for key in ('DESI_SPECTRO_REDUX', 'SPECPROD'):
            if key in cls.origenv:
                os.environ[key] = cls.origenv[key]
            elif key in os.environ:
                del os.environ[key]

    def tearDown(self):
        exptabdir = os.path.join(self.proddir, 'exposure_tables')
        if os.path.exists(exptabdir):
            shutil.rmtree(exptabdir)

    @staticmethod
    def _ptable(rows):
        """Build a minimal processing table from (jobdesc, status, qid) tuples"""
        return Table({
            'JOBDESC': [r[0] for r in rows],
            'STATUS': [r[1] for r in rows],
            'LATEST_QID': [r[2] for r in rows],
        })

    def _write_override(self, night, refnight, include):
        pathname = findfile('override', night=night)
        os.makedirs(os.path.dirname(pathname), exist_ok=True)
        with open(pathname, 'w') as fil:
            fil.write('calibration:\n    linkcal:\n'
                      + f'        refnight: {refnight}\n'
                      + f'        include: {include}\n')

    def test_no_table(self):
        """A missing or empty table provides nothing"""
        self.assertFalse(bias_dependency_available(None, 20230915))
        self.assertFalse(bias_dependency_available(self._ptable([]), 20230915))

    def test_submitted_biasnight(self):
        """A submitted biasnight is a usable dependency"""
        ptab = self._ptable([('biasnight', 'SUBMITTED', 5001)])
        self.assertTrue(bias_dependency_available(ptab, 20230915))

    def test_submitted_biaspdark(self):
        """A combined biaspdark also provides the bias"""
        ptab = self._ptable([('biaspdark', 'SUBMITTED', 5001)])
        self.assertTrue(bias_dependency_available(ptab, 20230915))

    def test_completed_with_default_qid(self):
        """Outputs already on disk count: COMPLETED with the default qid"""
        ptab = self._ptable([('biasnight', 'COMPLETED', get_default_qid())])
        self.assertTrue(bias_dependency_available(ptab, 20230915))

    def test_unsubmitted_biasnight_rejected(self):
        """A biasnight whose submission failed provides nothing to depend on"""
        ptab = self._ptable([('biasnight', 'UNSUBMITTED', get_err_qid())])
        self.assertFalse(bias_dependency_available(ptab, 20230915))

    def test_states_needing_resubmission_rejected(self):
        """Any state the pipeline would resubmit cannot be depended on

        submit_biasnight_and_preproc_darks() returns an existing processing
        table untouched when a bias row is already present, so a row left in a
        failed state by an earlier run arrives here with a real queue id.
        """
        for state in get_resubmission_states():
            with self.subTest(state=state):
                ptab = self._ptable([('biasnight', state, 5001)])
                self.assertFalse(bias_dependency_available(ptab, 20230915))

    def test_running_and_pending_accepted(self):
        """In-flight jobs are fine to depend on; Slurm handles the ordering"""
        for state in ('SUBMITTED', 'PENDING', 'RUNNING', 'COMPLETED'):
            with self.subTest(state=state):
                ptab = self._ptable([('biasnight', state, 5001)])
                self.assertTrue(bias_dependency_available(ptab, 20230915))

    def test_error_qid_rejected(self):
        """The error qid means the job never made it into the queue"""
        ptab = self._ptable([('biasnight', 'SUBMITTED', get_err_qid())])
        self.assertFalse(bias_dependency_available(ptab, 20230915))

    def test_linkcal_accepted_when_it_links_biasnight(self):
        """A linkcal supplies the bias if this night links biasnight onward"""
        self._write_override(20230915, 20230910, include='biasnight')
        ptab = self._ptable([('linkcal', 'SUBMITTED', 5001)])
        self.assertTrue(bias_dependency_available(ptab, 20230915))

    def test_linkcal_rejected_when_it_links_something_else(self):
        """A linkcal for another calibration type does not supply a bias

        Reachable when the reference night has no zeros, so no bias job is
        created, while its own override links only e.g. fiberflatnight.
        """
        self._write_override(20230915, 20230910, include='fiberflatnight')
        ptab = self._ptable([('linkcal', 'SUBMITTED', 5001)])
        self.assertFalse(bias_dependency_available(ptab, 20230915))

    def test_unrelated_jobs_rejected(self):
        """Calibration jobs that are not bias-providing do not count"""
        ptab = self._ptable([('psfnight', 'SUBMITTED', 5001),
                             ('nightlyflat', 'SUBMITTED', 5002)])
        self.assertFalse(bias_dependency_available(ptab, 20230915))

    def test_good_row_alongside_failed_row(self):
        """One usable row is enough even if another failed"""
        ptab = self._ptable([('biasnight', 'UNSUBMITTED', get_err_qid()),
                             ('biaspdark', 'SUBMITTED', 5002)])
        self.assertTrue(bias_dependency_available(ptab, 20230915))


class TestSubmitEarlyRefnightCalibrations(unittest.TestCase):
    """
    Tests for submit_early_refnight_calibrations(), with both submission entry
    points mocked so nothing is submitted.

    This is the function that performs the out-of-order submissions, and the
    stage-A-before-stage-B ordering is the invariant the whole feature rests on.
    """

    @classmethod
    def setUpClass(cls):
        cls.reduxdir = tempfile.mkdtemp()
        cls.specprod = 'test'
        cls.proddir = os.path.join(cls.reduxdir, cls.specprod)
        cls.logpath = os.path.join(cls.proddir, 'run', 'logs')
        os.makedirs(cls.logpath)
        cls.origenv = os.environ.copy()
        os.environ['DESI_SPECTRO_REDUX'] = cls.reduxdir
        os.environ['SPECPROD'] = cls.specprod

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.reduxdir)
        for key in ('DESI_SPECTRO_REDUX', 'SPECPROD'):
            if key in cls.origenv:
                os.environ[key] = cls.origenv[key]
            elif key in os.environ:
                del os.environ[key]

    def setUp(self):
        """Patch both submission entry points and the log redirection"""
        patchers = [
            patch('desispec.scripts.submit_prod.'
                  'submit_necessary_biasnights_and_preproc_darks'),
            patch('desispec.scripts.submit_prod.proc_night'),
            patch('desispec.scripts.submit_prod.stdouterr_redirected',
                  _null_redirect),
        ]
        self.bias = patchers[0].start()
        self.pn = patchers[1].start()
        patchers[2].start()
        for patcher in patchers:
            self.addCleanup(patcher.stop)
        self.bias.return_value = self._good_bias_table()

    def tearDown(self):
        exptabdir = os.path.join(self.proddir, 'exposure_tables')
        if os.path.exists(exptabdir):
            shutil.rmtree(exptabdir)

    @staticmethod
    def _good_bias_table():
        return Table({'JOBDESC': ['biasnight'], 'STATUS': ['SUBMITTED'],
                      'LATEST_QID': [5001]})

    def _write_exptable(self, night):
        pathname = findfile('exposure_table', night=night)
        os.makedirs(os.path.dirname(pathname), exist_ok=True)
        open(pathname, 'w').close()

    def _write_override(self, night, refnight, include):
        pathname = findfile('override', night=night)
        os.makedirs(os.path.dirname(pathname), exist_ok=True)
        with open(pathname, 'w') as fil:
            fil.write('calibration:\n    linkcal:\n'
                      + f'        refnight: {refnight}\n'
                      + f'        include: {include}\n')

    def _setup(self, include='biasnight', night=20230914, refnight=20230915):
        self._write_exptable(night)
        self._write_exptable(refnight)
        self._write_override(night, refnight, include=include)
        return night, refnight

    def _run(self, nights, **kwargs):
        return submit_early_refnight_calibrations(
            nights=nights, logpath=self.logpath, **kwargs)

    # ==================================================================
    # ordering, which is the invariant the feature rests on
    # ==================================================================

    def _record_calls(self):
        """Record (stage, night) in call order across both stages"""
        calls = []

        def bias(*args, **kwargs):
            calls.append(('A', kwargs['reference_night']))
            return self._good_bias_table()

        def procnight(*args, **kwargs):
            calls.append(('B', kwargs['night']))

        self.bias.side_effect = bias
        self.pn.side_effect = procnight
        return calls

    def test_stage_a_precedes_stage_b(self):
        """The bias-only submission must happen before that night's proc_night"""
        night, refnight = self._setup(include='biasnight')
        calls = self._record_calls()
        submitted = self._run([night, refnight])
        self.assertEqual(submitted, [refnight])
        self.assertEqual(calls, [('A', refnight), ('B', refnight)])

    def test_stages_are_paired_per_night(self):
        """Two independent reference nights each get their own A then B

        The pairing is what matters: a night's own bias must precede its own
        darknight reach. A global 'all A then all B' split would instead give
        A,A,B,B and break chains.
        """
        for night in (20230914, 20230915, 20230924, 20230925):
            self._write_exptable(night)
        self._write_override(20230914, 20230915, include='biasnight')
        self._write_override(20230924, 20230925, include='biasnight')
        calls = self._record_calls()
        self._run([20230914, 20230915, 20230924, 20230925])
        self.assertEqual(calls, [('A', 20230915), ('B', 20230915),
                                 ('A', 20230925), ('B', 20230925)])

    def test_chained_override_submits_prerequisite_first(self):
        """A night's prerequisite is fully submitted before that night runs

        A links biasnight from the later night B, and B itself links
        fiberflatnight from the earlier night C. C must be submitted before
        anything is done for B, or B's linkcal would be created before the
        calibrations it links to exist.
        """
        for night in (20230910, 20230915, 20230920):
            self._write_exptable(night)
        self._write_override(20230915, 20230920, include='biasnight')
        self._write_override(20230920, 20230910, include='fiberflatnight')
        calls = self._record_calls()
        submitted = self._run([20230915, 20230920])
        ## C first (no bias needed of its own), then B's bias, then the rest of B
        self.assertEqual(calls, [('B', 20230910),
                                 ('A', 20230920), ('B', 20230920)])
        self.assertEqual(submitted, [20230910, 20230920])

    def test_stage_a_skipped_when_biasnight_not_linked(self):
        """Only a linked biasnight needs the standalone bias submission"""
        night, refnight = self._setup(include='fiberflatnight')
        submitted = self._run([night, refnight])
        self.bias.assert_not_called()
        self.assertEqual(self.pn.call_count, 1)
        self.assertEqual(submitted, [refnight])

    # ==================================================================
    # arguments forwarded to each stage
    # ==================================================================

    def test_obstypes_passed_to_each_stage(self):
        """Stage A requests only zeros; stage B requests everything but science"""
        night, refnight = self._setup(include='biasnight')
        self._run([night, refnight])
        self.assertEqual(self.bias.call_args.kwargs['proc_obstypes'], ['zero'])
        stage_b_obstypes = self.pn.call_args.kwargs['proc_obstypes']
        self.assertNotIn('science', stage_b_obstypes)
        for obstype in ('zero', 'dark', 'arc', 'flat'):
            self.assertIn(obstype, stage_b_obstypes)

    def test_queue_and_reservation_forwarded(self):
        """Both stages must honour the queue and reservation"""
        night, refnight = self._setup(include='biasnight')
        self._run([night, refnight], queue='realtime', reservation='RES1')
        for mock in (self.bias, self.pn):
            self.assertEqual(mock.call_args.kwargs['queue'], 'realtime')
            self.assertEqual(mock.call_args.kwargs['reservation'], 'RES1')

    def test_reference_night_passed_to_stage_a(self):
        """Stage A must run on the reference night, not the linking night"""
        night, refnight = self._setup(include='biasnight')
        self._run([night, refnight])
        self.assertEqual(self.bias.call_args.kwargs['reference_night'], refnight)
        self.assertEqual(self.pn.call_args.kwargs['night'], refnight)

    # ==================================================================
    # failure and idempotency paths
    # ==================================================================

    def test_no_bias_landing_raises(self):
        """If stage A produces no usable bias row, abort before stage B"""
        night, refnight = self._setup(include='biasnight')
        self.bias.return_value = Table({'JOBDESC': [], 'STATUS': [],
                                        'LATEST_QID': []})
        with self.assertRaises(RuntimeError):
            self._run([night, refnight])
        self.pn.assert_not_called()

    def test_failed_bias_submission_raises(self):
        """A bias row left UNSUBMITTED is not accepted as a dependency"""
        night, refnight = self._setup(include='biasnight')
        self.bias.return_value = Table({'JOBDESC': ['biasnight'],
                                        'STATUS': ['UNSUBMITTED'],
                                        'LATEST_QID': [get_err_qid()]})
        with self.assertRaises(RuntimeError):
            self._run([night, refnight])
        self.pn.assert_not_called()

    def test_existing_completed_bias_is_idempotent(self):
        """Re-running with the bias already done proceeds without error"""
        night, refnight = self._setup(include='biasnight')
        self.bias.return_value = Table({'JOBDESC': ['biasnight'],
                                        'STATUS': ['COMPLETED'],
                                        'LATEST_QID': [get_default_qid()]})
        submitted = self._run([night, refnight])
        self.assertEqual(submitted, [refnight])
        self.assertEqual(self.pn.call_count, 1)

    def test_dry_run_level_4_submits_nothing(self):
        """dry_run_level >= 4 must not call either submission entry point"""
        night, refnight = self._setup(include='biasnight')
        submitted = self._run([night, refnight], dry_run_level=4)
        self.bias.assert_not_called()
        self.pn.assert_not_called()
        self.assertEqual(submitted, [refnight])

    def test_nothing_to_do_returns_empty(self):
        """No forward-pointing override means no early submissions at all"""
        self._write_exptable(20230914)
        self._write_exptable(20230915)
        self._write_override(20230915, 20230914, include='biasnight')
        submitted = self._run([20230914, 20230915])
        self.assertEqual(submitted, [])
        self.bias.assert_not_called()
        self.pn.assert_not_called()


class TestQueueThresholdBlocksPrePass(unittest.TestCase):
    """
    Tests that a full queue cannot let submit_production() skip the pre-pass
    and carry on.

    The pre-pass is a prerequisite for the chronological loop, and that loop
    polls the queue again independently. If the pre-pass were merely skipped, a
    queue that drained between the two polls would let an override night be
    submitted with no reference night calibrations to link.
    """

    @classmethod
    def setUpClass(cls):
        cls.reduxdir = tempfile.mkdtemp()
        cls.specprod = 'test'
        cls.proddir = os.path.join(cls.reduxdir, cls.specprod)
        os.makedirs(os.path.join(cls.proddir, 'run'))
        cls.origenv = os.environ.copy()
        os.environ['DESI_SPECTRO_REDUX'] = cls.reduxdir
        os.environ['SPECPROD'] = cls.specprod
        cls.sentinel = os.path.join(cls.proddir, 'run',
                                    'prod_submission_complete.txt')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.reduxdir)
        for key in ('DESI_SPECTRO_REDUX', 'SPECPROD'):
            if key in cls.origenv:
                os.environ[key] = cls.origenv[key]
            elif key in os.environ:
                del os.environ[key]

    def setUp(self):
        patchers = [
            patch('desispec.scripts.submit_prod.'
                  'submit_necessary_biasnights_and_preproc_darks'),
            patch('desispec.scripts.submit_prod.proc_night'),
            patch('desispec.scripts.submit_prod.check_queue_count'),
            patch('desispec.scripts.submit_prod.stdouterr_redirected',
                  _null_redirect),
        ]
        self.bias = patchers[0].start()
        self.pn = patchers[1].start()
        self.queue = patchers[2].start()
        patchers[3].start()
        for patcher in patchers:
            self.addCleanup(patcher.stop)
        self.bias.return_value = Table({'JOBDESC': ['biasnight'],
                                        'STATUS': ['SUBMITTED'],
                                        'LATEST_QID': [5001]})

    def tearDown(self):
        for sub in ('exposure_tables', 'processing_tables'):
            path = os.path.join(self.proddir, sub)
            if os.path.exists(path):
                shutil.rmtree(path)
        if os.path.exists(self.sentinel):
            os.remove(self.sentinel)

    def _write_exptable(self, night):
        pathname = findfile('exposure_table', night=night)
        os.makedirs(os.path.dirname(pathname), exist_ok=True)
        open(pathname, 'w').close()

    def _write_override(self, night, refnight, include):
        pathname = findfile('override', night=night)
        os.makedirs(os.path.dirname(pathname), exist_ok=True)
        with open(pathname, 'w') as fil:
            fil.write('calibration:\n    linkcal:\n'
                      + f'        refnight: {refnight}\n'
                      + f'        include: {include}\n')

    def _config(self, nights):
        """Write a production yaml and return its pathname"""
        pathname = os.path.join(self.reduxdir, 'prod_config.yaml')
        with open(pathname, 'w') as fil:
            fil.write(f"SPECPROD: '{self.specprod}'\n")
            fil.write(f"NIGHTS: {list(nights)}\n")
            fil.write(f"THRU_NIGHT: {max(nights)}\n")
            fil.write("Z_SUBMIT_TYPES: 'false'\n")
        return pathname

    def test_queue_draining_after_a_skipped_pre_pass_must_not_proceed(self):
        """The whole invocation must stop, not just skip the pre-pass

        submit_production() polls the queue once before the pre-pass and again
        inside the chronological loop. This simulates the queue being full for
        the first poll and drained for the rest, which is the race that would
        otherwise let the loop submit an override night with no reference
        night calibrations to link.
        """
        self._write_exptable(20230914)
        self._write_exptable(20230915)
        self._write_override(20230914, 20230915, include='biasnight')
        ## full at the pre-pass check, then drained for every later poll
        self.queue.side_effect = [99999] + [0] * 20
        submit_production(self._config([20230914, 20230915]), queue_threshold=10)
        ## neither the pre-pass nor the chronological loop may have submitted
        self.bias.assert_not_called()
        self.pn.assert_not_called()
        ## and the sentinel must not be written, so a later run retries
        self.assertFalse(os.path.exists(self.sentinel))

    def test_full_queue_without_pre_pass_still_reaches_main_loop(self):
        """An ordinary production must not be newly blocked by a full queue

        With no forward-pointing override there is no prerequisite, so the
        existing per-night queue check in the main loop should decide, exactly
        as it did before: a queue that drains lets the nights be submitted.
        """
        self._write_exptable(20230914)
        self._write_exptable(20230915)
        self.queue.side_effect = [99999] + [0] * 20
        submit_production(self._config([20230914, 20230915]), queue_threshold=10)
        self.bias.assert_not_called()
        ## no prerequisite exists, so the drained queue is allowed to proceed
        self.assertEqual(self.pn.call_count, 2)

    def test_quiet_queue_runs_pre_pass_then_main_loop(self):
        """With room in the queue both the pre-pass and the loop should run"""
        self._write_exptable(20230914)
        self._write_exptable(20230915)
        self._write_override(20230914, 20230915, include='biasnight')
        self.queue.return_value = 0
        submit_production(self._config([20230914, 20230915]),
                          queue_threshold=4500)
        self.bias.assert_called_once()
        ## stage B for the reference night, plus both nights in the main loop
        self.assertEqual(self.pn.call_count, 3)
        self.assertTrue(os.path.exists(self.sentinel))


if __name__ == '__main__':
    unittest.main()
