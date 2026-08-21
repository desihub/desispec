# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.scripts.proc_night
"""

import os
import glob
import unittest
import tempfile
import shutil
import importlib
import yaml

from desispec.workflow.batch_writer import get_desi_proc_tilenight_batch_file_pathname
import numpy as np

import desispec.workflow.exptable
import desispec.workflow.proctable
from desispec.workflow.processing import update_and_recursively_submit
from desispec.workflow.tableio import load_table, write_table
from desispec.workflow.redshifts import get_ztile_script_pathname
from desispec.workflow.batch_writer import \
    get_desi_proc_batch_file_path
from desispec.io import findfile
from desispec.test.util import link_rawdata

from desispec.scripts.proc_night import proc_night
import desispec.scripts.tile_redshifts
from desiutil.log import get_logger

## directory with real raw data for testing at NERSC
_dailynight = 20230915
_real_rawdir = os.path.expandvars(f'$DESI_ROOT/spectro/data')
_real_rawnight_dir = os.path.join(_real_rawdir, str(_dailynight))

class TestProcNight(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.prenight = 20230913
        cls.night = 20230914
        cls.repeat_tiles = [7567, 23826]
        cls.dailynight = _dailynight
        cls.basicnight = 20211129  #- early data without 1s CTE flat or end-of-night zeros/darks
        cls.laternight = 20250318 #- later night with both CTE and darknight

        cls.reduxdir = tempfile.mkdtemp()
        cls.test_rawdir = tempfile.mkdtemp()
        cls.test_rawnight_dir = os.path.join(cls.test_rawdir, str(cls.dailynight))
        os.makedirs(cls.test_rawnight_dir)

        cls.real_rawdir = _real_rawdir
        cls.real_rawnight_dir = _real_rawnight_dir

        cls.specprod = 'test'
        cls.proddir = os.path.join(cls.reduxdir, cls.specprod)

        cls.origenv = os.environ.copy()
        os.environ['DESI_SPECTRO_REDUX'] = cls.reduxdir
        os.environ['DESI_SPECTRO_DATA'] = cls.test_rawdir
        os.environ['SPECPROD'] = cls.specprod
        os.environ['NERSC_HOST'] = 'perlmutter'  # pretend to be on Perlmutter for testing
        ### os.environ['DESI_LOGLEVEL'] = 'WARNING' # reduce output from all the proc_night calls

        os.makedirs(cls.proddir)
        expdir = importlib.resources.files('desispec').joinpath('test', 'data', 'exposure_tables')
        shutil.copytree(expdir, os.path.join(cls.proddir, 'exposure_tables'))

        cls.etable_file = findfile('exposure_table', cls.night)
        cls.etable = load_table(cls.etable_file)
        cls.override_file = findfile('override', cls.night) # these are created in function

    def tearDown(self):
        desispec.workflow.proctable.reset_tilenight_ptab_cache()
        # remove everything from prod except exposure_table for self.night
        for path in glob.glob(self.proddir+'/*'):
            if os.path.basename(path) == 'exposure_tables':
                pass
            elif os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)

        # remove override_file if leftover from failed test
        for night in (self.night, self.dailynight, self.basicnight):
            override_file = findfile('override', night=night)
            if os.path.isfile(override_file):
                os.remove(override_file)

        # remove rawdir/dailynight contents
        for explink in glob.glob(f'{self.test_rawnight_dir}/*'):
            os.remove(explink)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.reduxdir)
        shutil.rmtree(cls.test_rawdir)
        for key in ('DESI_SPECTRO_REDUX', 'SPECPROD', 'NERSC_HOST', 'DESI_LOGLEVEL'):
            if key in cls.origenv:
                os.environ[key] = cls.origenv[key]
            elif key in os.environ:
                del os.environ[key]

    def test_proc_night(self):
        proctable, unproctable = proc_night(self.night, z_submit_types=['cumulative',],
                                            dry_run_level=1, sub_wait_time=0.0)

        # processing table file created
        self.assertTrue(os.path.isfile(findfile('processing_table', self.night)))

        # every tile is represented
        self.assertEqual(set(self.etable['TILEID']), set(proctable['TILEID']))

        # every step is represented. Note arcs and flats are bundled into the
        # psfnight/nightlyflat jobs, so they have no rows of their own.
        for jobdesc in ('ccdcalib', 'psfnight', 'nightlyflat', 'cteflat', 'tilenight', 'cumulative'):
            self.assertIn(jobdesc, set(proctable['JOBDESC']))
        for jobdesc in ('arc', 'flat'):
            self.assertNotIn(jobdesc, set(proctable['JOBDESC']))

        # tilenight jobs created
        for tileid in np.unique(proctable['TILEID']):
            if tileid<0: continue
            batchscript = get_desi_proc_tilenight_batch_file_pathname(self.night, tileid) + '.slurm'
            self.assertTrue(os.path.exists(batchscript), f'Missing {batchscript}')

        # ztile jobs created
        ii = proctable['JOBDESC'] == 'cumulative'
        for prow in proctable[ii]:
            batchscript = get_ztile_script_pathname(tileid=prow['TILEID'], group='cumulative', night=self.night)
            self.assertTrue(os.path.exists(batchscript), f'Missing {batchscript}')

        # internal IDs are unique per row
        unique_intids = np.unique(proctable['INTID'])
        self.assertEqual(len(unique_intids), len(proctable))

    def test_proc_night_dryrun3(self):
        """Test that dry_run_level=3 doesn't produce any output"""
        proctable, unproctable = proc_night(self.night, z_submit_types=['cumulative',],
                                            dry_run_level=3, sub_wait_time=0.0)

        prodfiles = glob.glob(self.proddir+'/*')
        self.assertEqual(len(prodfiles), 1)
        self.assertTrue(prodfiles[0].endswith('exposure_tables'))

    def test_proc_night_dryrun4(self):
        """Test that dry_run_level=4 doesn't produce any output"""
        proctable, unproctable = proc_night(self.night, z_submit_types=['cumulative',],
                                            dry_run_level=4, sub_wait_time=0.0)

        prodfiles = glob.glob(self.proddir+'/*')
        self.assertEqual(len(prodfiles), 1)
        self.assertTrue(prodfiles[0].endswith('exposure_tables'))

    def test_proc_night_noz(self):
        """Test that z_submit_types=None doesn't submit any redshift jobs"""

        #- subset of tiles
        ntiles = 2
        tiles = np.unique(self.etable[self.etable['OBSTYPE']=='science']['TILEID'])[0:ntiles]

        proctable, unproctable = proc_night(self.night, z_submit_types=None,
                                            tiles=tiles,
                                            dry_run_level=1, sub_wait_time=0.0)

        #- tilenight but not zproc batch scripts exist
        for tileid in tiles:
            batchscript = get_desi_proc_tilenight_batch_file_pathname(self.night, tileid) + '.slurm'
            self.assertTrue(os.path.exists(batchscript), f'Missing {batchscript}')

            zbatchscript = get_ztile_script_pathname(tileid=tileid, group='cumulative', night=self.night)
            self.assertFalse(os.path.exists(zbatchscript), f'Unexpected {batchscript}')

        #- Check that only the subset of tiles were processed
        proctiles = proctable['TILEID'][proctable['OBSTYPE'] == 'science']
        self.assertEqual(len(np.unique(proctiles)), ntiles)

    def test_proc_night_cross_night_redshifts(self):
        """Test if crossnight redshifts are submitted properly."""
        proctable1, unproctable1 = proc_night(self.prenight, sub_wait_time=0.0, dry_run_level=1)
        desispec.workflow.exptable.reset_science_etab_cache()
        desispec.workflow.proctable.reset_tilenight_ptab_cache()
        proctable2, unproctable2 = proc_night(self.night, sub_wait_time=0.0,
                                              dry_run_level=1, z_submit_types=['cumulative'])

        ## Test that cumulative redshift has dependency on previous night's job
        ## as well as the tilenight job from the second night
        for tileid in self.repeat_tiles:
            tilematches1 = proctable1[proctable1['TILEID'] == tileid]
            tilenight1 = tilematches1[tilematches1['JOBDESC']=='tilenight'][0]
            tilematches2 = proctable2[proctable2['TILEID'] == tileid]
            tilenight2 = tilematches2[tilematches2['JOBDESC']=='tilenight'][0]
            cumulative2 = tilematches2[tilematches2['JOBDESC'] == 'cumulative'][0]

            self.assertTrue(len(cumulative2['INT_DEP_IDS']) == 2)
            self.assertTrue(tilenight1['INTID'] in cumulative2['INT_DEP_IDS'])
            self.assertTrue(tilenight2['INTID'] in cumulative2['INT_DEP_IDS'])

            scriptpath = get_ztile_script_pathname(tileid, group='cumulative',
                                                   night=self.night)
            with open(scriptpath, 'r') as fil:
                for line in fil.readlines():
                    if 'desi_zproc' in line:
                        self.assertTrue(str(self.prenight) in line)
                        self.assertTrue(str(tilenight1['EXPID'][0]) in line)
                        self.assertTrue(str(self.night) in line)
                        self.assertTrue(str(tilenight2['EXPID'][0]) in line)

    def test_proc_night_resubmit_queue_failures(self):
        """Test if crossnight redshifts work properly with desi_resubmit_queue_failures."""
        proctable1, unproctable1 = proc_night(self.prenight, sub_wait_time=0.0, dry_run_level=1)
        desispec.workflow.exptable.reset_science_etab_cache()
        desispec.workflow.proctable.reset_tilenight_ptab_cache()
        proctable2, unproctable2 = proc_night(self.night, sub_wait_time=0.0,
                                              dry_run_level=1, z_submit_types=['cumulative'])
        desispec.workflow.exptable.reset_science_etab_cache()
        desispec.workflow.proctable.reset_tilenight_ptab_cache()

        ## test that the code runs
        updatedtable2, nsubmits, nbad = update_and_recursively_submit(proctable2, submits=0, dry_run_level=4)
        self.assertFalse(np.any(np.isin(updatedtable2['STATUS'], [b'DEP_NOT_SUBD', b'TIMEOUT'])),
                        msg='No TIMEOUTs in nominal resubmission')

        ## now test that the resubmission works by forcing the failure in redshift job
        for tileid in self.repeat_tiles:
            tilematches2 = proctable2[proctable2['TILEID'] == tileid]
            cumulative2 = tilematches2[tilematches2['JOBDESC'] == 'cumulative'][0]
            proctable2['STATUS'][proctable2['INTID']==cumulative2['INTID']] = 'TIMEOUT'
        updatedtable2, nsubmits, nbad = update_and_recursively_submit(proctable2,
                                                                submits=0,
                                                                dry_run_level=4)
        self.assertFalse(np.any(np.isin(updatedtable2['STATUS'], [b'DEP_NOT_SUBD', b'TIMEOUT'])),
                        msg='Cross night resubmission should leave no TIMEOUTs')

        ## now set the tilenight from the earlier night as bad
        ## now resubmission should refuse to proceed
        ## Set earlier tilenight as TIMEOUT, along with redshift job as TIMEOUT
        for tileid in self.repeat_tiles:
            tilematches1 = proctable1[proctable1['TILEID'] == tileid]
            tilenight1 = tilematches1[tilematches1['JOBDESC'] == 'tilenight'][0]
            proctable1['STATUS'][proctable1['INTID'] == tilenight1['INTID']] = 'TIMEOUT'
            tilematches2 = proctable2[proctable2['TILEID'] == tileid]
            cumulative2 = tilematches2[tilematches2['JOBDESC'] == 'cumulative'][0]
            proctable2['STATUS'][proctable2['INTID']==cumulative2['INTID']] = 'TIMEOUT'

        ## Save the updated proctable so that the resubmission code finds it
        tablename = findfile('proctable', night=self.prenight)
        write_table(proctable1, tablename=tablename, tabletype='proctable')
        desispec.workflow.proctable.reset_full_ptab_cache()

        ## Run resubmission code
        updatedtable2, nsubmits, nbad = update_and_recursively_submit(proctable2,
                                                                submits=0,
                                                                dry_run_level=4)
        self.assertTrue(np.sum(updatedtable2['STATUS'] == 'DEP_NOT_SUBD')==2,
                        msg='Cross night resubmission should have 2 DEP_NOT_SUBDs' \
                            + ' after forcing failed previous night jobs.')


    def _bundle_row(self, proctable, jobdesc):
        """Return the single processing row with the given JOBDESC"""
        sel = proctable['JOBDESC'] == jobdesc
        self.assertEqual(np.sum(sel), 1,
                         f'expected exactly one {jobdesc} row')
        return proctable[sel][0]

    def _bundle_script(self, night, jobdesc):
        """Return the text of the one generated script for the given JOBDESC"""
        scriptdir = get_desi_proc_batch_file_path(night, reduxdir=self.proddir)
        scripts = glob.glob(os.path.join(scriptdir, f'{jobdesc}*.slurm'))
        self.assertEqual(len(scripts), 1, f'expected one {jobdesc} script')
        with open(scripts[0], 'r') as fil:
            return fil.read()

    def test_proc_night_calibration_bundles(self):
        """Arcs, normal flats, and CTE flats are each submitted as one job.

        A normal night should produce exactly one psfnight row holding every
        selected arc, one nightlyflat row holding every selected normal flat,
        and one cteflat row holding every selected CTE flat, with no individual
        per-exposure calibration rows at all.
        """
        night = self.laternight
        proctable, unproctable = proc_night(night, z_submit_types=None,
                                            tiles=[], dry_run_level=1,
                                            sub_wait_time=0.0)

        ## no individual arc or flat rows remain
        for jobdesc in ('arc', 'flat'):
            self.assertNotIn(jobdesc, set(proctable['JOBDESC']))

        arcbundle = self._bundle_row(proctable, 'psfnight')
        flatbundle = self._bundle_row(proctable, 'nightlyflat')
        ctebundle = self._bundle_row(proctable, 'cteflat')

        ## a normal night has 5 arcs, 12 flats, and 3 CTE flats
        self.assertEqual(len(arcbundle['EXPID']), 5)
        self.assertEqual(len(flatbundle['EXPID']), 12)
        self.assertEqual(len(ctebundle['EXPID']), 3)

        ## the bundles carry the OBSTYPE of their exposures
        self.assertEqual(arcbundle['OBSTYPE'], 'arc')
        self.assertEqual(flatbundle['OBSTYPE'], 'flat')
        self.assertEqual(ctebundle['OBSTYPE'], 'flat')

        ## CTE flats belong to the cteflat bundle, not the nightlyflat bundle
        self.assertEqual(len(set(ctebundle['EXPID']) & set(flatbundle['EXPID'])), 0)

        ## all three are calibrators with unique internal IDs
        for bundle in (arcbundle, flatbundle, ctebundle):
            self.assertEqual(bundle['CALIBRATOR'], 1)
        self.assertEqual(len(np.unique(proctable['INTID'])), len(proctable))

        ## dependencies: arcs on ccdcalib, flats on arcs, CTE on arcs
        ccdcalib = self._bundle_row(proctable, 'ccdcalib')
        self.assertEqual(list(arcbundle['INT_DEP_IDS']), [ccdcalib['INTID']])
        self.assertEqual(list(flatbundle['INT_DEP_IDS']), [arcbundle['INTID']])
        self.assertEqual(list(ctebundle['INT_DEP_IDS']), [arcbundle['INTID']])
        self.assertNotIn(flatbundle['INTID'], list(ctebundle['INT_DEP_IDS']))

        ## the temporary per-exposure step rows must never be referenced
        intids = set(np.array(proctable['INTID']))
        for prow in proctable:
            for depid in prow['INT_DEP_IDS']:
                self.assertIn(depid, intids,
                              f"dangling dependency {depid} in {prow['JOBDESC']}")

        ## each bundle recorded the script that was actually written
        scriptdir = get_desi_proc_batch_file_path(night, reduxdir=self.proddir)
        for bundle in (arcbundle, flatbundle, ctebundle):
            self.assertNotEqual(bundle['SCRIPTNAME'], '')
            self.assertTrue(os.path.exists(os.path.join(scriptdir,
                                                        bundle['SCRIPTNAME'])))

    def test_proc_night_bundle_dependency_of_tilenight(self):
        """Tilenight depends on the bundled nightlyflat job"""
        night = self.laternight
        proctable, unproctable = proc_night(night, z_submit_types=None,
                                            dry_run_level=3, sub_wait_time=0.0)
        flatbundle = self._bundle_row(proctable, 'nightlyflat')
        tnights = proctable[proctable['JOBDESC'] == 'tilenight']
        self.assertGreater(len(tnights), 0)
        for tnight in tnights:
            self.assertEqual(list(tnight['INT_DEP_IDS']), [flatbundle['INTID']])

    def test_proc_night_bundle_scripts(self):
        """The three generated bundle scripts have the required structure"""
        night = self.laternight
        proctable, unproctable = proc_night(night, z_submit_types=None,
                                            tiles=[], dry_run_level=1,
                                            sub_wait_time=0.0)
        arcbundle = self._bundle_row(proctable, 'psfnight')
        flatbundle = self._bundle_row(proctable, 'nightlyflat')
        ctebundle = self._bundle_row(proctable, 'cteflat')

        arctext = self._bundle_script(night, 'psfnight')
        flattext = self._bundle_script(night, 'nightlyflat')
        ctetext = self._bundle_script(night, 'cteflat')

        ## one exposure command per EXPID. Arcs and CTE flats echo the command
        ## before running it, so they appear twice; flats are run by parallel,
        ## whose -v flag does the echoing, so they appear once.
        for text, bundle, ncopies in ((arctext, arcbundle, 2),
                                      (flattext, flatbundle, 1),
                                      (ctetext, ctebundle, 2)):
            for expid in bundle['EXPID']:
                cmd = (f"desi_proc --cameras {bundle['PROCCAMWORD']}"
                       + f' -n {night} -e {expid} --mpi')
                self.assertEqual(text.count(cmd), ncopies,
                                 f'unexpected number of commands for {expid}')

        ## exactly one joint fit for arcs and flats (echoed then run)
        self.assertEqual(arctext.count('desi_proc_joint_fit --obstype arc'), 2)
        self.assertEqual(flattext.count('desi_proc_joint_fit --obstype flat'), 2)

        ## the nightlyflat joint fit lists only the normal flats
        expid_str = ','.join([str(e) for e in flatbundle['EXPID']])
        self.assertIn(f'-e {expid_str} --mpi', flattext)
        for expid in ctebundle['EXPID']:
            self.assertNotIn(str(expid), expid_str)

        ## arcs are backgrounded and waited on individually
        self.assertEqual(arctext.count(' &\npids="$pids $!"'), 5)
        self.assertIn('wait $pid || nfail=$((nfail+1))', arctext)
        self.assertNotIn('\nwait\n', arctext)

        ## flats use GNU parallel throttled by the allocation
        self.assertIn('parallel -v -j "$SLURM_JOB_NUM_NODES"', flattext)
        self.assertIn("STARTTIMESTR='--starttime $(date +%s)'", flattext)
        self.assertIn('nfail=$?', flattext)
        self.assertIn('echo FAILED to process $nfail individual flats', flattext)

        ## CTE flats are serial, have no joint fit, and accumulate failures
        self.assertNotIn('parallel', ctetext)
        self.assertNotIn(' &\n', ctetext)
        self.assertNotIn('desi_proc_joint_fit', ctetext)
        self.assertIn('nfail=$((nfail+1))', ctetext)
        for expid in ctebundle['EXPID']:
            self.assertIn(f'-e {expid} --mpi', ctetext)

        ## every command runs directly under MPI, and no exposure is abandoned
        ## because a sibling failed
        for text in (arctext, flattext, ctetext):
            self.assertNotIn('--batch', text)
            self.assertNotIn('--nosubmit', text)
            self.assertNotIn('--halt', text)
            self.assertNotIn('kill ', text)
            for line in text.split('\n'):
                if line.startswith('srun ') or line.startswith('"srun '):
                    self.assertIn(' --mpi ', line)

    def test_proc_night_cteflat_idempotency(self):
        """Re-running proc_night doesn't submit a second CTE bundle"""
        night = self.laternight
        proctable1, unproc1 = proc_night(night, z_submit_types=None, tiles=[],
                                         dry_run_level=1, sub_wait_time=0.0)
        self.assertEqual(np.sum(proctable1['JOBDESC'] == 'cteflat'), 1)

        proctable2, unproc2 = proc_night(night, z_submit_types=None, tiles=[],
                                         dry_run_level=1, sub_wait_time=0.0)
        self.assertEqual(np.sum(proctable2['JOBDESC'] == 'cteflat'), 1)
        self.assertEqual(len(proctable2), len(proctable1))

    def test_proc_night_cteflat_with_linked_calibrations(self):
        """A night with every standard calibration accounted for still processes CTE flats.

        The cteflat bundle is independent of the standard calibrations: its
        preproc images feed the nightly detector QA, not fiberflatnight, so
        linking psfnight and fiberflatnight from a reference night must not
        suppress it.
        """
        night = self.laternight
        ## link every linkable calibration and skip darknight, so that every
        ## accounted_for flag is True and the CTE flats are the only work left
        override_dict = {'calibration':
                            {'linkcal':
                                {'refnight': night - 1}}}
        proctable, unproctable = self._override_write_run_delete(
                override_dict, night=night, tiles=[], z_submit_types=None,
                no_darknight=True, dry_run_level=1)

        self.assertIn('linkcal', set(proctable['JOBDESC']))
        ## all the standard calibrations are accounted for by the link
        for jobdesc in ('psfnight', 'nightlyflat', 'ccdcalib', 'biasnight',
                        'biaspdark', 'pdark'):
            self.assertNotIn(jobdesc, set(proctable['JOBDESC']))
        ## but the CTE flats still need to be processed
        ctebundle = self._bundle_row(proctable, 'cteflat')
        self.assertEqual(len(ctebundle['EXPID']), 3)
        ## with nothing else to depend on, it depends on the linkcal job
        linkcal = self._bundle_row(proctable, 'linkcal')
        self.assertEqual(list(ctebundle['INT_DEP_IDS']), [linkcal['INTID']])

    def test_proc_night_failed_cteflat_blocks_science(self):
        """A failed CTE bundle is caught by the calibrator-failure check.

        The cteflat row has CALIBRATOR=1, so an unrecoverable failure must stop
        proc_night from submitting more work unless the user explicitly passes
        ignore_proc_table_failures.
        """
        from unittest.mock import patch

        night = self.laternight
        proctable, unproctable = proc_night(night, z_submit_types=None,
                                            tiles=[], dry_run_level=1,
                                            sub_wait_time=0.0)
        ## Mark the cteflat bundle as failed and save it back to disk
        ctesel = proctable['JOBDESC'] == 'cteflat'
        self.assertEqual(proctable['CALIBRATOR'][ctesel][0], 1)
        proctable['STATUS'][ctesel] = 'TIMEOUT'
        write_table(proctable, tablename=findfile('processing_table', night),
                    tabletype='proctable')

        ## Pretend that resubmitting the failure didn't fix it, which is what
        ## happens once a job has used up its resubmission attempts.
        with patch('desispec.scripts.proc_night.update_from_queue',
                   side_effect=lambda ptab, **kwargs: ptab), \
             patch('desispec.scripts.proc_night.update_and_recursively_submit',
                   side_effect=lambda ptab, **kwargs: (ptab, 0, 1)):
            with self.assertRaises(AssertionError):
                proc_night(night, z_submit_types=None, tiles=[],
                           dry_run_level=4, sub_wait_time=0.0)

            ## the documented override lets the user proceed anyway
            proctable2, unproctable2 = proc_night(
                    night, z_submit_types=None, tiles=[], dry_run_level=4,
                    sub_wait_time=0.0, ignore_proc_table_failures=True)
        self.assertEqual(np.sum(proctable2['JOBDESC'] == 'cteflat'), 1)

    def _touch_camera_files(self, filetype, night, expid=None, cameras=None):
        """Create empty per-camera output files so restarts can find them"""
        if cameras is None:
            cameras = [f'{band}{spectro}' for spectro in range(10)
                       for band in 'brz']
        for camera in cameras:
            pathname = findfile(filetype, night=night, expid=expid,
                                camera=camera)
            os.makedirs(os.path.dirname(pathname), exist_ok=True)
            open(pathname, 'w').close()

    def test_proc_night_bundle_restarts(self):
        """Existing products prune bundle steps or suppress bundles entirely"""
        night = self.laternight
        scriptdir = get_desi_proc_batch_file_path(night, reduxdir=self.proddir)

        ## First find out which exposures the bundles would use
        proctable, unproctable = proc_night(night, z_submit_types=None,
                                            tiles=[], dry_run_level=3,
                                            sub_wait_time=0.0)
        arc_expids = list(self._bundle_row(proctable, 'psfnight')['EXPID'])
        cte_expids = list(self._bundle_row(proctable, 'cteflat')['EXPID'])

        ## An existing psfnight for every camera suppresses the arc bundle,
        ## and existing fitpsf files drop the finished arcs from the script
        self._touch_camera_files('psfnight', night)
        for expid in arc_expids[:2]:
            self._touch_camera_files('fitpsf', night, expid=expid)
        ## Existing CTE frames drop the finished CTE exposures
        self._touch_camera_files('frame', night, expid=cte_expids[0])

        proctable, unproctable = proc_night(night, z_submit_types=None,
                                            tiles=[], dry_run_level=1,
                                            sub_wait_time=0.0)

        ## psfnight is already on disk, so no arc bundle script was written
        arcbundle = self._bundle_row(proctable, 'psfnight')
        self.assertEqual(arcbundle['STATUS'], 'COMPLETED')
        self.assertEqual(arcbundle['SCRIPTNAME'], '',
                         'a bundle completed before script generation must '
                         'not name a script that was never written')
        self.assertEqual(len(glob.glob(os.path.join(scriptdir, 'psfnight*.slurm'))), 0)

        ## the CTE bundle still runs, but only for the unfinished exposures
        ctetext = self._bundle_script(night, 'cteflat')
        self.assertNotIn(f'-e {cte_expids[0]} ', ctetext)
        for expid in cte_expids[1:]:
            self.assertIn(f'-e {expid} --mpi', ctetext)
        ## and its walltime shrank with the number of remaining steps
        self.assertIn('#SBATCH --time=00:40:00', ctetext)

    def test_proc_night_bundle_resubmission_pathname(self):
        """A bundle can be resubmitted using the SCRIPTNAME it recorded"""
        from desispec.workflow.processing import batch_script_pathname

        night = self.laternight
        proctable, unproctable = proc_night(night, z_submit_types=None,
                                            tiles=[], dry_run_level=1,
                                            sub_wait_time=0.0)

        for jobdesc in ('psfnight', 'nightlyflat', 'cteflat'):
            bundle = self._bundle_row(proctable, jobdesc)
            ## the stored name is nonempty and is what resubmission rebuilds
            self.assertNotEqual(bundle['SCRIPTNAME'], '')
            pathname = batch_script_pathname(bundle)
            self.assertEqual(os.path.basename(pathname), bundle['SCRIPTNAME'])
            self.assertTrue(os.path.exists(pathname), f'missing {pathname}')

        ## a failed bundle is resubmitted rather than left behind
        proctable['STATUS'][proctable['JOBDESC'] == 'cteflat'] = 'TIMEOUT'
        updated, nsubmits, nbad = update_and_recursively_submit(
                proctable, submits=0, dry_run_level=4)
        self.assertEqual(nbad, 0)
        self.assertGreater(nsubmits, 0)
        self.assertNotIn('TIMEOUT', set(updated['STATUS']))

    def _override_write_run_delete(self, override_dict, night=None, **kwargs):
        """Write override, run proc_night, remove override file, and return outputs"""
        desispec.workflow.proctable.reset_tilenight_ptab_cache()

        if night is None:
            night = self.night

        override_file = findfile('override', night=night)

        with open(override_file, 'w') as fil:
            yaml.safe_dump(override_dict, fil)
        proctable, unproctable = proc_night(night, sub_wait_time=0.0, **kwargs)
        os.remove(override_file)
        return proctable, unproctable

    def test_proc_night_linking_and_ccdcalib_earlynight(self):
        """Test if override file linking is working"""
        ## Setup the basic dictionary for the override file
        base_override_dict = {'calibration':
                                {'linkcal':
                                    {'refnight': self.night-1}}}

        ## Test basic case where we link everything
        with self.subTest(i=0):
            testdict = base_override_dict.copy()
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'biaspdark', 'psfnight', 'nightlyflat']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test no psfnight but still fiberflatnight -- should raise error
        with self.subTest(i=1):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'psfnight'
            with open(self.override_file, 'w') as fil:
                yaml.safe_dump(testdict, fil)
            with self.assertRaises(ValueError):
                proctable, unproctable = proc_night(self.night, sub_wait_time=0.0,
                                                    dry_run_level=3)
            os.remove(self.override_file)

        ## Test no psfnight but still fiberflatnight and flag set to allow
        with self.subTest(i=2):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'psfnight'
            proctable, unproctable = self._override_write_run_delete(testdict,
                                                                    dry_run_level=3,
                                                                    psf_linking_without_fflat=True)
            for job in ['linkcal', 'biasnight', 'ccdcalib', 'nightlyflat']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biaspdark', 'pdark', 'psfnight']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link fiberflatnight
        with self.subTest(i=3):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'fiberflatnight'
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'biasnight', 'ccdcalib', 'psfnight']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biaspdark', 'pdark', 'nightlyflat']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link biasnight
        with self.subTest(i=4):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'biasnight'
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'ccdcalib', 'psfnight', 'nightlyflat']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'biaspdark', 'pdark']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link badcolumns
        with self.subTest(i=5):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'badcolumns'
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'biasnight', 'ccdcalib', 'psfnight', 'nightlyflat']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biaspdark', 'pdark']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link ctecorrnight
        with self.subTest(i=6):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'ctecorrnight'
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'biasnight', 'ccdcalib', 'psfnight', 'nightlyflat']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biaspdark', 'pdark']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link ctecorrnight and biasnight
        with self.subTest(i=7):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'ctecorrnight,biasnight'
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'ccdcalib', 'psfnight', 'nightlyflat']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'biaspdark', 'pdark']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link badcolumns and biasnight
        with self.subTest(i=8):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'badcolumns,biasnight'
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'ccdcalib', 'psfnight', 'nightlyflat']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'biaspdark', 'pdark']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link cte and badcol
        with self.subTest(i=9):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'ctecorrnight,badcolumns'
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'biasnight', 'psfnight', 'nightlyflat']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biaspdark', 'pdark', 'ccdcalib']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link ccdcalib
        with self.subTest(i=10):
            calib_files = 'biasnight,badcolumns,ctecorrnight'
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = calib_files
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'psfnight', 'nightlyflat']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'pdark', 'biaspdark', 'ccdcalib']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link psfnight and fiberflatnight
        with self.subTest(i=11):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'psfnight,fiberflatnight'
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'biasnight', 'ccdcalib']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biaspdark', 'pdark', 'psfnight', 'nightlyflat']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link everything except fiberflatnight -- should raise error
        with self.subTest(i=12):
            calib_files = 'biasnight,badcolumns,ctecorrnight,psfnight'
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = calib_files
            with open(self.override_file, 'w') as fil:
                yaml.safe_dump(testdict, fil)
            with self.assertRaises(ValueError):
                proctable, unproctable = proc_night(self.night, sub_wait_time=0.0,
                                                    dry_run_level=3)
            os.remove(self.override_file)

        ## Test link everything except fiberflatnight with flag set to allow
        with self.subTest(i=13):
            calib_files = 'biasnight,badcolumns,ctecorrnight,psfnight'
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = calib_files
            proctable, unproctable = self._override_write_run_delete(testdict,
                                                                    dry_run_level=3,
                                                                    psf_linking_without_fflat=True)
            for job in ['linkcal', 'nightlyflat']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'biaspdark', 'pdark', 'ccdcalib', 'psfnight']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link everything except psfnight
        with self.subTest(i=14):
            calib_files = 'biasnight,badcolumns,ctecorrnight,fiberflatnight'
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = calib_files
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'psfnight']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'biaspdark', 'pdark', 'ccdcalib', 'nightlyflat']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test linking an earlier night without 1s CTE flat
        ## linking biasnight,badcolumns so ccdcalib should not be linked
        with self.subTest(i=15):
            testdict = base_override_dict.copy()
            testnight = self.basicnight
            testdict['calibration']['linkcal']['refnight'] = testnight-1
            testdict['calibration']['linkcal']['include'] = 'biasnight,badcolumns'
            proctable, unproctable = self._override_write_run_delete(testdict, night=testnight, dry_run_level=3)
            for job in ['linkcal', 'psfnight', 'nightlyflat', 'tilenight']:
                self.assertIn(job, set(set(proctable['JOBDESC'])))
            for job in ['biasnight', 'biaspdark', 'pdark', 'ccdcalib']:
                self.assertNotIn(job, set(set(proctable['JOBDESC'])))


    def test_proc_night_linking_and_ccdcalib_latenight(self):
        """Test if override file linking is working"""
        ## Setup the basic dictionary for the override file
        orig_night = self.night
        orig_override_file = self.override_file
        self.night = self.laternight
        self.override_file = findfile('override', night=self.laternight)

        base_override_dict = {'calibration':
                                {'linkcal':
                                    {'refnight': self.night-1}}}

        ## Test basic case where we link everything
        with self.subTest(i=0):
            testdict = base_override_dict.copy()
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'biaspdark', 'psfnight', 'nightlyflat']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test no psfnight but still fiberflatnight -- should raise error
        with self.subTest(i=1):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'psfnight'
            with open(self.override_file, 'w') as fil:
                yaml.safe_dump(testdict, fil)
            with self.assertRaises(ValueError):
                proctable, unproctable = proc_night(self.night, sub_wait_time=0.0,
                                                    dry_run_level=3)
            os.remove(self.override_file)

        ## Test no psfnight but still fiberflatnight and flag set to allow
        with self.subTest(i=2):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'psfnight'
            proctable, unproctable = self._override_write_run_delete(testdict,
                                                                    dry_run_level=3,
                                                                    psf_linking_without_fflat=True)
            for job in ['linkcal', 'biaspdark', 'ccdcalib', 'nightlyflat']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'pdark', 'psfnight']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link fiberflatnight
        with self.subTest(i=3):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'fiberflatnight'
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'biaspdark', 'ccdcalib', 'psfnight']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'pdark', 'nightlyflat']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link biasnight
        with self.subTest(i=4):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'biasnight'
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'pdark', 'ccdcalib', 'psfnight', 'nightlyflat']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'biaspdark']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link badcolumns
        with self.subTest(i=5):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'badcolumns'
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'biaspdark', 'ccdcalib', 'psfnight', 'nightlyflat']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'pdark']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link ctecorrnight
        with self.subTest(i=6):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'ctecorrnight'
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'biaspdark', 'ccdcalib', 'psfnight', 'nightlyflat']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'pdark']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link ctecorrnight and biasnight
        with self.subTest(i=7):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'ctecorrnight,biasnight'
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'pdark', 'ccdcalib', 'psfnight', 'nightlyflat']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'biaspdark']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link badcolumns and biasnight
        with self.subTest(i=8):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'badcolumns,biasnight'
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'pdark', 'ccdcalib', 'psfnight', 'nightlyflat']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'biaspdark']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link cte and badcol
        with self.subTest(i=9):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'ctecorrnight,badcolumns'
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'biaspdark', 'ccdcalib', 'psfnight', 'nightlyflat']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'pdark']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link ccdcalib
        with self.subTest(i=10):
            calib_files = 'biasnight,badcolumns,ctecorrnight'
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = calib_files
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'pdark', 'ccdcalib', 'psfnight', 'nightlyflat']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'biaspdark']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link psfnight and fiberflatnight
        with self.subTest(i=11):
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = 'psfnight,fiberflatnight'
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'biaspdark', 'ccdcalib']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'pdark', 'psfnight', 'nightlyflat']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link everything except fiberflatnight -- should raise error
        with self.subTest(i=12):
            calib_files = 'biasnight,badcolumns,ctecorrnight,psfnight'
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = calib_files
            with open(self.override_file, 'w') as fil:
                yaml.safe_dump(testdict, fil)
            with self.assertRaises(ValueError):
                proctable, unproctable = proc_night(self.night, sub_wait_time=0.0,
                                                    dry_run_level=3)
            os.remove(self.override_file)

        ## Test link everything except fiberflatnight with flag set to allow
        with self.subTest(i=13):
            calib_files = 'biasnight,badcolumns,ctecorrnight,psfnight'
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = calib_files
            proctable, unproctable = self._override_write_run_delete(testdict,
                                                                    dry_run_level=3,
                                                                    psf_linking_without_fflat=True)
            for job in ['linkcal', 'pdark', 'ccdcalib', 'nightlyflat']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'biaspdark', 'psfnight']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Test link everything except psfnight
        with self.subTest(i=14):
            calib_files = 'biasnight,badcolumns,ctecorrnight,fiberflatnight'
            testdict = base_override_dict.copy()
            testdict['calibration']['linkcal']['include'] = calib_files
            proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
            for job in ['linkcal', 'pdark', 'ccdcalib', 'psfnight']:
                self.assertIn(job, set(proctable['JOBDESC']))
            for job in ['biasnight', 'biaspdark', 'nightlyflat']:
                self.assertNotIn(job, set(proctable['JOBDESC']))

        ## Clean up by resetting override file and night to original values
        self.night = orig_night
        self.override_file = orig_override_file

    def test_proc_night_no_darknight(self):
        """Regression test for issue #2623: no_darknight=True must not trigger
        surrounding-night biasnight submissions.

        When no_darknight=True, the obstypes used for submitting surrounding-
        night dark-related jobs must exclude 'dark' so that
        get_stacked_dark_exposure_table is never called and only the reference
        night receives a biasnight job (no pdark or biaspdark).
        """
        from unittest.mock import patch

        with patch('desispec.workflow.submission.get_stacked_dark_exposure_table') as mock_get_stacked:
            proctable, unproctable = proc_night(self.laternight,
                                                no_darknight=True,
                                                dry_run_level=3,
                                                sub_wait_time=0.0)
            ## get_stacked_dark_exposure_table must NOT be called when do_darknight=False
            mock_get_stacked.assert_not_called()

        ## biasnight should be submitted for the reference night
        self.assertIn('biasnight', set(proctable['JOBDESC']))
        ## pdark and biaspdark must NOT be submitted when no_darknight=True
        for job in ['pdark', 'biaspdark']:
            self.assertNotIn(job, set(proctable['JOBDESC']))

    def test_proc_night_camword_linking(self):
        """Test if setting camword in override file linking is working"""
        ## Setup the basic dictionary for the override file
        base_override_dict = {'calibration':
                                {'linkcal':
                                    {'refnight': self.night-1}}}

        ## Test basic case where we link everything
        testdict = base_override_dict.copy()
        proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
        procrow = proctable[proctable['JOBDESC']=='linkcal']
        self.assertEqual(procrow['PROCCAMWORD'], 'a0123456789')

        ## Test custom camword
        testdict = base_override_dict.copy()
        testdict['calibration']['linkcal']['camword'] = 'a012'
        proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
        procrow = proctable[proctable['JOBDESC']=='linkcal']
        self.assertEqual(procrow['PROCCAMWORD'], 'a012')

    def test_proc_night_override_flag_setting(self):
        """Test if override file linking is working"""
        ## Setup the basic dictionary for the override file
        base_override_dict = {'calibration': {}}

        ## Test if flag appears when we request it
        testdict = base_override_dict.copy()
        flag = "--autocal-ff-solve-grad"
        testdict['calibration']['nightlyflat'] = {'extra_cmd_args': [flag]}
        proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=1)
        for job in ['ccdcalib', 'psfnight', 'nightlyflat', 'tilenight']:
            self.assertIn(job, set(proctable['JOBDESC']))
        for job in ['linkcal', 'nightlybias']:
            self.assertNotIn(job, set(proctable['JOBDESC']))
        scriptdir = get_desi_proc_batch_file_path(self.night, reduxdir=self.proddir)
        script = glob.glob(os.path.join(scriptdir, 'nightlyflat*.slurm'))[0]
        with open(script, 'r') as fil:
            for line in fil.readlines():
                if 'desi_proc_joint_fit' in line:
                    self.assertTrue(flag in line)
        ## Remove outputs of the last dry-run-level=1
        if os.path.isdir(scriptdir):
            shutil.rmtree(scriptdir)
        proctabledir = os.path.dirname(findfile('proctable', night=self.night))
        if os.path.isdir(proctabledir):
            shutil.rmtree(proctabledir)

        ## Now check that it doesn't have that string if we don't specify it
        flag = "--autocal-ff-solve-grad"
        testdict['calibration'] = {}
        proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=1)
        for job in ['ccdcalib', 'psfnight', 'nightlyflat', 'tilenight']:
            self.assertIn(job, set(proctable['JOBDESC']))
        for job in ['linkcal', 'nightlybias']:
            self.assertNotIn(job, set(proctable['JOBDESC']))
        script = glob.glob(os.path.join(scriptdir, 'nightlyflat*.slurm'))[0]
        with open(script, 'r') as fil:
            for line in fil.readlines():
                if 'desi_proc_joint_fit' in line:
                    self.assertFalse(flag in line)


    @unittest.skipIf('SKIP_PROC_NIGHT_DAILY_TEST' in os.environ, 'Skipping test_proc_night_daily because $SKIP_PROC_NIGHT_DAILY_TEST is set')
    @unittest.skipUnless(os.path.isdir(_real_rawnight_dir), f'{_real_rawnight_dir} not available')
    def test_proc_night_daily(self):
        """
        Test proc_night daily mode on nights with partial data

        Requires being at NERSC to inspect input raw data
        """

        while True:
            num_newlinks = link_rawdata(self.real_rawnight_dir, self.test_rawnight_dir, numexp=10)
            desispec.workflow.exptable.reset_science_etab_cache()
            desispec.workflow.proctable.reset_tilenight_ptab_cache()
            if num_newlinks == 0:
                break
            else:
                proctable, unproctable = proc_night(self.dailynight, daily=True, still_acquiring=True,
                                                    z_submit_types=['cumulative',],
                                                    dry_run_level=1, sub_wait_time=0.0)


                etable = load_table(findfile('exposure_table', self.dailynight))
                keep = etable['LASTSTEP'] != 'ignore'
                etable = etable[keep]

                ## if 1sec flat has arrived, cals should be submitted, otherwise nothing processed yet
                has_1secflat = np.any((etable['OBSTYPE']=='flat') & (np.abs(etable['EXPTIME']-1)<0.1))
                should_submit_biaspdark = (len(etable) > 2
                                           and np.sum(etable['OBSTYPE']=='dark') > 0
                                           and np.sum(etable['OBSTYPE']=='arc') > 0
                                           and np.sum(etable['OBSTYPE']=='zero') > 9)

                if has_1secflat:
                    ## if 1sec flat has arrived, cals should be submitted.
                    ## Note: this could be different if we switch to testing a daily night with
                    ## and override file, in which case e.g. it could have linkcal instead of nightlyflat
                    ## Note arcs and flats are bundled into the psfnight/nightlyflat/cteflat
                    ## jobs, so they have no rows of their own.
                    for jobdesc in ('biasnight', 'ccdcalib', 'psfnight',
                                    'nightlyflat', 'cteflat'):
                        self.assertIn(jobdesc, set(proctable['JOBDESC']))
                    for jobdesc in ('arc', 'flat'):
                        self.assertNotIn(jobdesc, set(proctable['JOBDESC']))
                elif should_submit_biaspdark:
                    ## arc have started coming in and there are biases and darks, so we expect
                    ## the biasnight or biaspdark job to be submitted
                    ## since this is a 2023 night, the afternoon 300s dark isn't used for darks,
                    ## so there are no darks to process and we expect a biasnight
                    self.assertEqual(len(proctable), 1)
                    self.assertIn('biasnight', set(proctable['JOBDESC']))
                else:
                    self.assertEqual(len(proctable), 0)

                ## count science tiles processed
                if np.any(etable['OBSTYPE'] == 'science'):
                    proctiles = set(proctable['TILEID'][ proctable['OBSTYPE'] == 'science' ])
                    exptiles = set(etable['TILEID'][ etable['OBSTYPE'] == 'science' ])

                    ## if last exposure is a science, we should not have processed that tile yet
                    ## since still_acquiring=True means we'll wait for more data from that tile
                    if etable['OBSTYPE'][-1] == 'science':
                        self.assertEqual(len(proctiles), len(exptiles)-1)
                    ## otherwise we've moved on to non-science, and will have processed all tiles
                    else:
                        self.assertEqual(len(proctiles), len(exptiles))


        ## Final pass with still_acquiring=False to finish last tile
        proctable, unproctable = proc_night(self.dailynight, daily=True, still_acquiring=False,
                                                z_submit_types=['cumulative',],
                                                dry_run_level=1, sub_wait_time=0.0)
        proctiles = set(proctable['TILEID'][ proctable['OBSTYPE'] == 'science' ])
        exptiles = set(etable['TILEID'][ etable['OBSTYPE'] == 'science' ])
        self.assertEqual(len(proctiles), len(exptiles))


def _make_biaspdark_ptable(night, extra_jobdescs=None):
    """Create a processing table with a biaspdark row and optional extra rows.

    Args:
        night (int): Observation night in YYYYMMDD format.
        extra_jobdescs (list of str or None): Additional JOBDESC values to
            append after the biaspdark row.  Default is None (no extras).

    Returns:
        Table: Processing table with one completed row per JOBDESC.
    """
    from desispec.workflow.proctable import (
        default_prow,
        instantiate_processing_table,
    )
    jobdescs = ['biaspdark'] + (extra_jobdescs or [])
    ptable = instantiate_processing_table()
    for intid, jobdesc in enumerate(jobdescs, start=1):
        prow = default_prow()
        prow['INTID'] = intid
        prow['JOBDESC'] = jobdesc
        prow['NIGHT'] = night
        prow['STATUS'] = 'COMPLETED'
        prow['EXPID'] = np.array([10000 + intid], dtype=int)
        prow['PROCCAMWORD'] = 'a0123456789'
        ptable.add_row(prow)
    return ptable


class TestSubmitFutureBiaspdarks(unittest.TestCase):
    """Regression tests for the submit_future_biaspdarks logic in proc_night.

    The submit_future_biaspdarks flag (issue #2697) is True when all of the
    following hold:
    - 'dark' is in the biaspdark obstypes
    - still_acquiring is False
    - 'ccdcalib' is NOT in the processing table
    - 'tilenight' is NOT in the processing table

    When True, proc_night must call submit_necessary_biasnights_and_preproc_darks
    without explicit n_nights constraints (i.e. with the default full-range
    kwargs).  When False (e.g. because ccdcalib is present), the function is
    still called via the submit_pdark path, but restricted to the current night
    only (n_nights_before=0, n_nights_after=0).
    """

    # Night after 20240510 (darknight enabled) and after 20211130 (cte flats enabled)
    _NIGHT = 20240601
    # Obstypes that include both 'zero' (for bias logic) and 'dark' (for pdark logic)
    _PROC_OBSTYPES = np.array(['zero', 'dark', 'flat', 'arc', 'science'])

    @classmethod
    def setUpClass(cls):
        cls.reduxdir = tempfile.mkdtemp()
        cls.specprod = 'test'
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
        self.tmp_dir = tempfile.mkdtemp()
        # proc_night checks that the exposure table file exists when
        # update_exptable=False (the default), so create an empty placeholder.
        self.exp_table_path = os.path.join(self.tmp_dir, 'exposure_table.csv')
        open(self.exp_table_path, 'w').close()
        self.proc_table_path = os.path.join(self.tmp_dir, 'processing_table.csv')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _run_proc_night_with_mocks(self, init_ptable):
        """Call proc_night with the heavy internals mocked out.

        Sets up patches needed to exercise the biaspdark submission decision
        logic without touching the filesystem or Slurm.

        Args:
            init_ptable (Table): Initial processing table returned by the
                load_tables mock.

        Returns:
            MagicMock: The mock for submit_necessary_biasnights_and_preproc_darks
                so the caller can assert on its call arguments.
        """
        from unittest.mock import patch
        from desispec.workflow.exptable import instantiate_exposure_table

        etable = instantiate_exposure_table()

        with patch('desispec.scripts.proc_night.load_tables',
                   return_value=(etable, init_ptable)), \
             patch('desispec.scripts.proc_night.submit_necessary_biasnights_and_preproc_darks',
                   return_value=init_ptable) as mock_submit, \
             patch('desispec.scripts.proc_night.update_from_queue',
                   return_value=init_ptable), \
             patch('desispec.scripts.proc_night.any_jobs_need_resubmission',
                   return_value=False), \
             patch('desispec.scripts.proc_night.load_override_file',
                   return_value={}), \
             patch('desispec.scripts.proc_night.generate_calibration_dict',
                   return_value={'accounted_for': []}), \
             patch('desispec.scripts.proc_night.all_calibs_submitted',
                   return_value=True), \
             patch('desispec.scripts.proc_night.determine_science_to_proc',
                   return_value=(etable, [])), \
             patch('desispec.scripts.proc_night.get_tiles_cumulative',
                   return_value=[]):
            proc_night(
                night=self._NIGHT,
                proc_obstypes=self._PROC_OBSTYPES,
                z_submit_types=None,
                exp_table_pathname=self.exp_table_path,
                proc_table_pathname=self.proc_table_path,
                dry_run_level=4,
                still_acquiring=False,
                no_darknight=False,
            )

        return mock_submit

    def test_future_biaspdarks_submitted_when_biaspdark_exists_without_ccdcalib(self):
        """submit_necessary_biasnights_and_preproc_darks is called via the
        future-nights path when biaspdark exists but ccdcalib and tilenight are absent.

        Regression test for issue #2697: a biaspdark row in the proctable that
        was submitted for a previous night's darknight run must not suppress
        the submission of future-night biaspdark jobs.  When ccdcalib and
        tilenight are absent, submit_future_biaspdarks is True and the function
        is called without explicit n_nights_before / n_nights_after constraints
        (i.e. with the default full-range kwargs).
        """
        init_ptable = _make_biaspdark_ptable(self._NIGHT)

        mock_submit = self._run_proc_night_with_mocks(init_ptable)

        mock_submit.assert_called_once()
        # Via the submit_future_biaspdarks path the call uses default
        # (full-range) kwargs, so n_nights_before/n_nights_after are absent.
        call_kwargs = mock_submit.call_args[1]
        self.assertNotIn('n_nights_before', call_kwargs)
        self.assertNotIn('n_nights_after', call_kwargs)

    def test_future_biaspdarks_not_triggered_when_ccdcalib_present(self):
        """When ccdcalib is present alongside biaspdark, submit_future_biaspdarks
        is False and the function is called via submit_pdark with n_nights=0.

        This verifies the negative case: submit_future_biaspdarks is only True
        when ccdcalib is absent.  When ccdcalib is present the code falls back
        to the submit_pdark path, which restricts the search to the current
        night only (n_nights_before=0, n_nights_after=0).
        """
        init_ptable = _make_biaspdark_ptable(self._NIGHT, extra_jobdescs=['ccdcalib'])

        mock_submit = self._run_proc_night_with_mocks(init_ptable)

        mock_submit.assert_called_once()
        call_kwargs = mock_submit.call_args[1]
        self.assertEqual(call_kwargs.get('n_nights_before'), 0)
        self.assertEqual(call_kwargs.get('n_nights_after'), 0)

    def test_future_biaspdarks_not_triggered_when_tilenight_present(self):
        """When tilenight is present alongside biaspdark, submit_future_biaspdarks
        is False and only the submit_pdark path is used.
        """
        init_ptable = _make_biaspdark_ptable(self._NIGHT, extra_jobdescs=['tilenight'])

        mock_submit = self._run_proc_night_with_mocks(init_ptable)

        mock_submit.assert_called_once()
        call_kwargs = mock_submit.call_args[1]
        self.assertEqual(call_kwargs.get('n_nights_before'), 0)
        self.assertEqual(call_kwargs.get('n_nights_after'), 0)
