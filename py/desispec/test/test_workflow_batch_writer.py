"""
Test desispec.workflow.batch_writer calibration bundle scripts
"""

import os
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

from desispec.workflow.batch import max_nodes_for_jobdesc
from desispec.workflow.batch_writer import \
    create_calibration_bundle_batch_script, \
    get_calibration_bundle_step_resources
from desispec.workflow.processing import desi_proc_command, \
    desi_proc_joint_fit_command
from desispec.workflow.proctable import default_prow

## Nights, exposures, and camwords of the hand-written reference scripts in
## evals/. Keep these in sync with evals/make_templates.py.
ARC_NIGHT = 20260706
ARC_EXPIDS = [360185, 360186, 360187, 360188, 360189]
FLAT_NIGHT = 20260806
FLAT_EXPIDS = [364647, 364648, 364649, 364652, 364653, 364654,
               364657, 364658, 364659, 364662, 364663, 364664]
CTE_NIGHT = 20260806
CTE_EXPIDS = [364665, 364666, 364667]
CTE_CAMWORDS = ['a0123456789', 'a012345678', 'a0123456789']
FULL_CAMWORD = 'a0123456789'
REFERENCE_REDUXDIR = '/global/cfs/cdirs/desi/spectro/redux/jobbundle'

_evals_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), 'evals')


def make_prow(night, expids, camword, obstype, jobdesc, badamps=''):
    """Create a minimal processing row for command generation"""
    prow = default_prow()
    prow['NIGHT'] = night
    prow['EXPID'] = list(expids)
    prow['PROCCAMWORD'] = camword
    prow['OBSTYPE'] = obstype
    prow['JOBDESC'] = jobdesc
    prow['BADAMPS'] = badamps
    return prow


def make_steps(night, expids, camwords, obstype='arc', step_jobdesc='arc',
               badamps=None, use_specter=False):
    """Create the bundle step dictionaries for the given exposures"""
    if badamps is None:
        badamps = [''] * len(expids)
    steps = []
    for expid, camword, amps in zip(expids, camwords, badamps):
        step_prow = make_prow(night, [expid], camword, obstype, step_jobdesc,
                              badamps=amps)
        steps.append({'expid': expid, 'camword': camword,
                      'cmd': desi_proc_command(step_prow, system_name=None,
                                               use_specter=use_specter,
                                               direct_mode=True)})
    return steps


class TestCalibrationBundleBatchScript(unittest.TestCase):
    """Tests for create_calibration_bundle_batch_script"""

    @classmethod
    def setUpClass(cls):
        cls._cached_nersc_host = os.getenv('NERSC_HOST')
        os.environ['NERSC_HOST'] = 'perlmutter'

    @classmethod
    def tearDownClass(cls):
        if cls._cached_nersc_host is None:
            if 'NERSC_HOST' in os.environ:
                del os.environ['NERSC_HOST']
        else:
            os.environ['NERSC_HOST'] = cls._cached_nersc_host

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write(self, **kwargs):
        """Generate a bundle script and return its text"""
        kwargs.setdefault('queue', 'realtime')
        scriptpathname = create_calibration_bundle_batch_script(
                reduxdir=self.tmpdir, **kwargs)
        with open(scriptpathname) as fx:
            return fx.read()

    def _arc_script(self, expids=None, camwords=None, **kwargs):
        expids = ARC_EXPIDS if expids is None else expids
        camwords = [FULL_CAMWORD] * len(expids) if camwords is None else camwords
        bundle = make_prow(ARC_NIGHT, expids, FULL_CAMWORD, 'arc', 'psfnight')
        return self._write(night=ARC_NIGHT, jobdesc='psfnight', expids=expids,
                           camword=FULL_CAMWORD,
                           steps=make_steps(ARC_NIGHT, expids, camwords),
                           joint_cmd=desi_proc_joint_fit_command(
                                   bundle, direct_mode=True), **kwargs)

    def _flat_script(self, expids=None, camwords=None, **kwargs):
        expids = FLAT_EXPIDS if expids is None else expids
        camwords = [FULL_CAMWORD] * len(expids) if camwords is None else camwords
        bundle = make_prow(FLAT_NIGHT, expids, FULL_CAMWORD, 'flat',
                           'nightlyflat')
        return self._write(night=FLAT_NIGHT, jobdesc='nightlyflat',
                           expids=expids, camword=FULL_CAMWORD,
                           steps=make_steps(FLAT_NIGHT, expids, camwords,
                                            obstype='flat',
                                            step_jobdesc='flat'),
                           joint_cmd=desi_proc_joint_fit_command(
                                   bundle, direct_mode=True), **kwargs)

    def _cte_script(self, expids=None, camwords=None, **kwargs):
        expids = CTE_EXPIDS if expids is None else expids
        camwords = CTE_CAMWORDS if camwords is None else camwords
        return self._write(night=CTE_NIGHT, jobdesc='cteflat', expids=expids,
                           camword=FULL_CAMWORD,
                           steps=make_steps(CTE_NIGHT, expids, camwords,
                                            obstype='flat',
                                            step_jobdesc='flat'), **kwargs)

    @staticmethod
    def _nodes(text):
        return int(re.search(r'#SBATCH -N (\d+)', text).group(1))

    @staticmethod
    def _walltime(text):
        hh, mm, ss = re.search(r'#SBATCH --time=(\d+):(\d+):(\d+)',
                               text).groups()
        return int(hh) * 60 + int(mm), int(ss)

    def test_step_resources(self):
        """Every bundle exposure step runs on a single node"""
        nodes, ntasks, threads, runtime = get_calibration_bundle_step_resources(
                'ARC', 30, system_name='perlmutter-cpu')
        self.assertEqual(nodes, 1)
        ## Schedule() consumes ranks in groups of 20 plus one scheduler rank
        self.assertEqual(ntasks % 20, 1)
        self.assertEqual(ntasks, 101)
        self.assertEqual(threads, 2)

        for jobclass in ('FLAT', 'CTEFLAT'):
            nodes, ntasks, threads, runtime = get_calibration_bundle_step_resources(
                    jobclass, 30, system_name='perlmutter-gpu')
            self.assertEqual(nodes, 1)
            self.assertEqual(ntasks, 64)
            self.assertEqual(threads, 2)

    def test_allocation_nodes(self):
        """Allocations hold one node per concurrent exposure step"""
        self.assertEqual(self._nodes(self._arc_script()), 5)
        self.assertEqual(self._nodes(self._flat_script()), 4)
        self.assertEqual(self._nodes(self._cte_script()), 1)

        ## degenerate flat sequences still need one node
        for nflats in (1, 2):
            text = self._flat_script(expids=FLAT_EXPIDS[:nflats])
            self.assertEqual(self._nodes(text), 1)

        ## CTE flats are serial regardless of how many there are
        for ncte in (1, 2, 3):
            text = self._cte_script(expids=CTE_EXPIDS[:ncte],
                                    camwords=CTE_CAMWORDS[:ncte])
            self.assertEqual(self._nodes(text), 1)

    def test_no_step_exceeds_the_allocation(self):
        """No individual srun asks for more nodes than the job requested"""
        for text in (self._arc_script(), self._flat_script(),
                     self._cte_script()):
            nodes = self._nodes(text)
            for match in re.finditer(r'srun [^\n"]*?-N (\d+)', text):
                self.assertLessEqual(int(match.group(1)), nodes)

    def test_walltimes(self):
        """Arc and flat walltimes cover the passes plus the joint fit"""
        ## one arc pass of 45 minutes plus an 8 minute psfnight
        arc_minutes, seconds = self._walltime(self._arc_script())
        self.assertEqual(arc_minutes, 45 + 8)
        self.assertEqual(seconds, 0)

        ## three passes of 20 minutes plus an 8 minute nightlyflat
        flat_minutes, _ = self._walltime(self._flat_script())
        self.assertEqual(flat_minutes, 3 * 20 + 8)

        ## the CTE walltime is the sum over its serial steps
        for ncte in (1, 2, 3):
            cte_minutes, _ = self._walltime(
                    self._cte_script(expids=CTE_EXPIDS[:ncte],
                                     camwords=CTE_CAMWORDS[:ncte]))
            self.assertEqual(cte_minutes, ncte * 20)

        ## minutes are always below 60 in the HH:MM:SS request
        for text in (self._arc_script(), self._flat_script(),
                     self._cte_script()):
            hh, mm, ss = re.search(r'#SBATCH --time=(\d+):(\d+):(\d+)',
                                   text).groups()
            self.assertLess(int(mm), 60)

    def test_flat_concurrency_override(self):
        """A non-default flat concurrency changes the allocation and walltime"""
        text = self._flat_script(concurrency=2)
        self.assertEqual(self._nodes(text), 2)
        minutes, _ = self._walltime(text)
        self.assertEqual(minutes, 6 * 20 + 8)

    def test_forced_runtime(self):
        """An explicit runtime overrides the estimate"""
        text = self._arc_script(runtime=90)
        self.assertEqual(self._walltime(text)[0], 90)

    def test_arc_request_above_cap_warns_and_clamps(self):
        """An oversized arc bundle reduces concurrency instead of raising"""
        expids = list(range(360185, 360185 + 20))
        with patch('desispec.workflow.batch_writer.get_logger') as mock_logger:
            text = self._arc_script(expids=expids)
            warnings = [str(call) for call
                        in mock_logger.return_value.warning.call_args_list]
        self.assertTrue(any('multiple passes' in msg for msg in warnings),
                        'expected a warning that the arcs run in passes')
        cap = max_nodes_for_jobdesc('PSFNIGHT')
        self.assertEqual(self._nodes(text), cap)
        ## 20 arcs over 15 nodes needs 2 passes plus the joint fit
        self.assertEqual(self._walltime(text)[0], 2 * 45 + 8)

    def test_arc_exposure_block(self):
        """Arcs are all backgrounded, then each PID is waited on"""
        text = self._arc_script()
        self.assertEqual(text.count(' &\npids="$pids $!"'), len(ARC_EXPIDS))
        self.assertIn('for pid in $pids; do', text)
        self.assertIn('wait $pid || nfail=$((nfail+1))', text)
        ## a bare wait would always return 0 and is a regression
        self.assertNotIn('\nwait\n', text)
        ## the gate sits before the joint fit
        gate = text.index('not running psfnight')
        self.assertLess(gate, text.index('desi_proc_joint_fit'))
        ## OMP_NUM_THREADS is exported once, then reset before psfnight
        self.assertEqual(text.count('export OMP_NUM_THREADS'), 2)
        self.assertLess(text.index('export OMP_NUM_THREADS=2'),
                        text.index('export OMP_NUM_THREADS=1'))
        self.assertLess(text.index('export OMP_NUM_THREADS=1'),
                        text.index('desi_proc_joint_fit'))

    def test_flat_exposure_block(self):
        """Flats are throttled by GNU parallel with no per-wave barrier"""
        text = self._flat_script()
        self.assertIn('parallel -v -j "$SLURM_JOB_NUM_NODES"', text)
        self.assertEqual(text.count('"srun '), len(FLAT_EXPIDS))
        self.assertNotIn('--halt', text)
        ## STARTTIMESTR must stay a literal so each job stamps its own time
        self.assertIn("STARTTIMESTR='--starttime $(date +%s)'", text)
        self.assertIn('${STARTTIMESTR}', text)
        ## parallel's exit status is captured before it can be clobbered
        self.assertIn('nfail=$?', text)
        self.assertLess(text.index('nfail=$?'), text.index('if [ $nfail -ne 0 ]'))
        self.assertIn('echo FAILED to process $nfail individual flats', text)
        self.assertLess(text.index('echo FAILED to process $nfail'),
                        text.index('desi_proc_joint_fit'))

    def test_cte_exposure_block(self):
        """CTE flats run serially and accumulate their failures"""
        text = self._cte_script()
        self.assertNotIn('parallel', text)
        self.assertNotIn('desi_proc_joint_fit', text)
        self.assertNotIn(' &\n', text)
        ## no exit between exposure steps, so a failure doesn't stop the rest
        body = text[text.index('nfail=0'):]
        self.assertEqual(body.count('exit 1'), 1)
        self.assertEqual(body.count('nfail=$((nfail+1))'), len(CTE_EXPIDS))
        self.assertIn('echo FAILED: $nfail of 3 CTE exposures failed', text)
        ## each exposure keeps its own camword
        for expid, camword in zip(CTE_EXPIDS, CTE_CAMWORDS):
            self.assertIn(f'-e {expid} --mpi', text)
            self.assertIn(f'cteflat-{CTE_NIGHT}-{expid:08d}-{camword}-timing', text)

    def test_per_step_files_are_distinct(self):
        """Each exposure step gets its own timing file and log file"""
        for text, night, expids, camwords, stepdesc in (
                (self._arc_script(), ARC_NIGHT, ARC_EXPIDS,
                 [FULL_CAMWORD] * len(ARC_EXPIDS), 'arc'),
                (self._flat_script(), FLAT_NIGHT, FLAT_EXPIDS,
                 [FULL_CAMWORD] * len(FLAT_EXPIDS), 'flat'),
                (self._cte_script(), CTE_NIGHT, CTE_EXPIDS, CTE_CAMWORDS,
                 'cteflat')):
            timingfiles = set(re.findall(r'--timingfile (\S+)', text))
            logfiles = set(re.findall(r'> (\S+\.log)', text))
            ## one per exposure, plus the joint fit for arcs and flats
            njoint = 0 if stepdesc == 'cteflat' else 1
            self.assertEqual(len(timingfiles), len(expids) + njoint)
            self.assertEqual(len(logfiles), len(expids))
            for expid, camword in zip(expids, camwords):
                base = f'{stepdesc}-{night}-{expid:08d}-{camword}'
                self.assertTrue(any(t.startswith(base + '-timing')
                                    for t in timingfiles), base)
                self.assertTrue(any(l.startswith(base + '-')
                                    for l in logfiles), base)

    def test_gpu_options(self):
        """GPU exposure steps request GPUs; the GPU joint step inherits them"""
        flattext = self._flat_script()
        for line in flattext.split('\n'):
            if 'desi_proc_joint_fit' in line and line.startswith('srun'):
                self.assertIn('desi_mps_wrapper', line)
                self.assertNotIn('--gpus-per-node', line)
            elif '"srun ' in line:
                self.assertIn('--gpus-per-node=4', line)
                self.assertIn('desi_mps_wrapper', line)
        self.assertIn('export MPICH_GPU_SUPPORT_ENABLED=1', flattext)
        self.assertIn('#SBATCH --account desi_g', flattext)

        ## arcs run on CPUs and use neither
        arctext = self._arc_script()
        self.assertNotIn('desi_mps_wrapper', arctext)
        self.assertNotIn('MPICH_GPU_SUPPORT_ENABLED', arctext)
        self.assertIn('#SBATCH --constraint=cpu', arctext)
        self.assertIn('#SBATCH --account desi', arctext)

    def test_per_exposure_camwords_and_badamps(self):
        """Per exposure camwords and badamps reach the generated commands"""
        camwords = ['a0123456789', 'a012345678', 'a0123456789', 'a01234567',
                    'a0123456789']
        badamps = ['', 'b7D', '', '', 'z8A']
        bundle = make_prow(ARC_NIGHT, ARC_EXPIDS, FULL_CAMWORD, 'arc',
                           'psfnight')
        text = self._write(night=ARC_NIGHT, jobdesc='psfnight',
                           expids=ARC_EXPIDS, camword=FULL_CAMWORD,
                           steps=make_steps(ARC_NIGHT, ARC_EXPIDS, camwords,
                                            badamps=badamps),
                           joint_cmd=desi_proc_joint_fit_command(
                                   bundle, direct_mode=True))
        for expid, camword, amps in zip(ARC_EXPIDS, camwords, badamps):
            self.assertIn(f'--cameras {camword} -n {ARC_NIGHT} -e {expid}', text)
            if amps:
                self.assertIn(f'-e {expid} --mpi --badamps={amps}', text)

    def test_use_specter_preserved_for_flats(self):
        """use-specter reaches the individual flat commands"""
        bundle = make_prow(FLAT_NIGHT, FLAT_EXPIDS, FULL_CAMWORD, 'flat',
                           'nightlyflat')
        text = self._write(night=FLAT_NIGHT, jobdesc='nightlyflat',
                           expids=FLAT_EXPIDS, camword=FULL_CAMWORD,
                           steps=make_steps(FLAT_NIGHT, FLAT_EXPIDS,
                                            [FULL_CAMWORD] * len(FLAT_EXPIDS),
                                            obstype='flat', step_jobdesc='flat',
                                            use_specter=True),
                           joint_cmd=desi_proc_joint_fit_command(
                                   bundle, direct_mode=True))
        self.assertEqual(text.count('--use-specter'), len(FLAT_EXPIDS))

    def test_joint_only_script(self):
        """A bundle whose exposures are all done still runs its joint fit"""
        bundle = make_prow(ARC_NIGHT, ARC_EXPIDS, FULL_CAMWORD, 'arc',
                           'psfnight')
        text = self._write(night=ARC_NIGHT, jobdesc='psfnight',
                           expids=ARC_EXPIDS, camword=FULL_CAMWORD, steps=[],
                           joint_cmd=desi_proc_joint_fit_command(
                                   bundle, direct_mode=True))
        self.assertEqual(text.count('desi_proc --cameras'), 0)
        self.assertEqual(text.count('desi_proc_joint_fit'), 2)
        ## the allocation is still valid for the joint fit alone
        self.assertGreaterEqual(self._nodes(text), 1)

    def test_direct_execution_commands(self):
        """Every command runs directly under MPI"""
        for text in (self._arc_script(), self._flat_script(),
                     self._cte_script()):
            self.assertNotIn('--batch', text)
            self.assertNotIn('--nosubmit', text)
            for line in text.split('\n'):
                if line.startswith('srun ') or line.startswith('"srun '):
                    self.assertIn(' --mpi', line)

    def test_invalid_bundles_rejected(self):
        """Mismatched joint commands and unknown descriptors raise"""
        steps = make_steps(CTE_NIGHT, CTE_EXPIDS, CTE_CAMWORDS,
                           obstype='flat', step_jobdesc='flat')
        with self.assertRaises(ValueError):
            ## cteflat has no joint fit
            self._write(night=CTE_NIGHT, jobdesc='cteflat', expids=CTE_EXPIDS,
                        camword=FULL_CAMWORD, steps=steps,
                        joint_cmd='desi_proc_joint_fit --obstype flat')
        with self.assertRaises(ValueError):
            ## psfnight requires one
            self._write(night=ARC_NIGHT, jobdesc='psfnight', expids=ARC_EXPIDS,
                        camword=FULL_CAMWORD, steps=steps)
        with self.assertRaises(ValueError):
            self._write(night=ARC_NIGHT, jobdesc='flat', expids=ARC_EXPIDS,
                        camword=FULL_CAMWORD, steps=steps)


@unittest.skipUnless(os.path.isdir(_evals_dir),
                     f'{_evals_dir} reference scripts not available')
class TestCalibrationBundleTemplates(unittest.TestCase):
    """Compare generated bundle scripts to the reference scripts in evals/

    These golden files document exactly what the writer is expected to emit.
    Regenerate them with evals/make_templates.py when the writer changes
    intentionally.
    """

    @classmethod
    def setUpClass(cls):
        cls._cached_nersc_host = os.getenv('NERSC_HOST')
        os.environ['NERSC_HOST'] = 'perlmutter'

    @classmethod
    def tearDownClass(cls):
        if cls._cached_nersc_host is None:
            if 'NERSC_HOST' in os.environ:
                del os.environ['NERSC_HOST']
        else:
            os.environ['NERSC_HOST'] = cls._cached_nersc_host

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _compare(self, template, **kwargs):
        scriptpathname = create_calibration_bundle_batch_script(
                queue='realtime', reduxdir=self.tmpdir, **kwargs)
        with open(scriptpathname) as fx:
            generated = fx.read().replace(self.tmpdir, REFERENCE_REDUXDIR)
        with open(os.path.join(_evals_dir, template)) as fx:
            expected = fx.read()
        self.assertEqual(generated, expected,
                         f'{template} is out of date; regenerate it with '
                         + 'evals/make_templates.py')

    def test_arc_template(self):
        """The arc bundle reproduces evals/arc_job_template.slurm"""
        bundle = make_prow(ARC_NIGHT, ARC_EXPIDS, FULL_CAMWORD, 'arc',
                           'psfnight')
        self._compare('arc_job_template.slurm', night=ARC_NIGHT,
                      jobdesc='psfnight', expids=ARC_EXPIDS,
                      camword=FULL_CAMWORD,
                      steps=make_steps(ARC_NIGHT, ARC_EXPIDS,
                                       [FULL_CAMWORD] * len(ARC_EXPIDS)),
                      joint_cmd=desi_proc_joint_fit_command(bundle,
                                                            direct_mode=True))

    def test_flat_template(self):
        """The normal flat bundle reproduces evals/flat_job_template.slurm"""
        bundle = make_prow(FLAT_NIGHT, FLAT_EXPIDS, FULL_CAMWORD, 'flat',
                           'nightlyflat')
        self._compare('flat_job_template.slurm', night=FLAT_NIGHT,
                      jobdesc='nightlyflat', expids=FLAT_EXPIDS,
                      camword=FULL_CAMWORD,
                      steps=make_steps(FLAT_NIGHT, FLAT_EXPIDS,
                                       [FULL_CAMWORD] * len(FLAT_EXPIDS),
                                       obstype='flat', step_jobdesc='flat'),
                      joint_cmd=desi_proc_joint_fit_command(bundle,
                                                            direct_mode=True))

    def test_cteflat_template(self):
        """The CTE flat bundle reproduces evals/cteflat_job_template.slurm"""
        self._compare('cteflat_job_template.slurm', night=CTE_NIGHT,
                      jobdesc='cteflat', expids=CTE_EXPIDS,
                      camword=FULL_CAMWORD,
                      steps=make_steps(CTE_NIGHT, CTE_EXPIDS, CTE_CAMWORDS,
                                       obstype='flat', step_jobdesc='flat'))


if __name__ == '__main__':
    unittest.main()
