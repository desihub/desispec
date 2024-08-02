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

import numpy as np

import desispec.workflow.exptable
import desispec.workflow.proctable
from desispec.workflow.processing import update_and_recursively_submit
from desispec.workflow.tableio import load_table, write_table
from desispec.workflow.redshifts import get_ztile_script_pathname
from desispec.workflow.desi_proc_funcs import \
    get_desi_proc_tilenight_batch_file_pathname, get_desi_proc_batch_file_path
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

        # every step is represented
        for jobdesc in ('ccdcalib', 'arc', 'psfnight', 'flat', 'nightlyflat', 'tilenight', 'cumulative'):
            self.assertIn(jobdesc, proctable['JOBDESC'])

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
        updatedtable2, nsubmits = update_and_recursively_submit(proctable2, submits=0, dry_run=3)
        self.assertFalse(np.any(np.in1d(updatedtable2['STATUS'], [b'DEP_NOT_SUBD', b'TIMEOUT'])),
                        msg='No TIMEOUTs in nominal resubmission')

        ## now test that the resubmission works by forcing the failure in redshift job
        for tileid in self.repeat_tiles:
            tilematches2 = proctable2[proctable2['TILEID'] == tileid]
            cumulative2 = tilematches2[tilematches2['JOBDESC'] == 'cumulative'][0]
            proctable2['STATUS'][proctable2['INTID']==cumulative2['INTID']] = 'TIMEOUT'
        updatedtable2, nsubmits = update_and_recursively_submit(proctable2,
                                                                submits=0,
                                                                dry_run=1)
        self.assertFalse(np.any(np.in1d(updatedtable2['STATUS'], [b'DEP_NOT_SUBD', b'TIMEOUT'])),
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
        updatedtable2, nsubmits = update_and_recursively_submit(proctable2,
                                                                submits=0,
                                                                dry_run=1)
        self.assertTrue(np.any(np.in1d(updatedtable2['STATUS'], [b'DEP_NOT_SUBD', b'TIMEOUT'])),
                        msg='Cross night resubmission should leave two TIMEOUTs')
        self.assertTrue(np.sum(updatedtable2['STATUS'] == 'DEP_NOT_SUBD')==2,
                        msg='Cross night resubmission should have 2 TIMEOUTs' \
                            + ' after forcing failed previous night jobs.')


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

    def test_proc_night_linking_and_ccdcalib(self):
        """Test if override file linking is working"""
        ## Setup the basic dictionary for the override file
        base_override_dict = {'calibration':
                                {'linkcal':
                                    {'refnight': self.night-1}}}

        ## Test basic case where we link everything
        testdict = base_override_dict.copy()
        proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
        for job in ['linkcal']:
            self.assertTrue(job in proctable['JOBDESC'])
        for job in ['nightlybias', 'ccdcalib', 'psfnight', 'nightlyflat']:
            self.assertTrue(job not in proctable['JOBDESC'])

        ## Test no psfnight but still fiberflatnight -- should raise error
        testdict = base_override_dict.copy()
        testdict['calibration']['linkcal']['include'] = 'psfnight'
        with open(self.override_file, 'w') as fil:
            yaml.safe_dump(testdict, fil)
        with self.assertRaises(ValueError):
            proctable, unproctable = proc_night(self.night, sub_wait_time=0.0,
                                                dry_run_level=3)
        os.remove(self.override_file)

        ## Test no psfnight but still fiberflatnight and flag set to allow
        testdict = base_override_dict.copy()
        testdict['calibration']['linkcal']['include'] = 'psfnight'
        proctable, unproctable = self._override_write_run_delete(testdict,
                                                                 dry_run_level=3,
                                                                 psf_linking_without_fflat=True)
        for job in ['linkcal', 'ccdcalib', 'nightlyflat']:
            self.assertTrue(job in proctable['JOBDESC'])
        for job in ['nightlybias', 'psfnight']:
            self.assertTrue(job not in proctable['JOBDESC'])

        ## Test link fiberflatnight
        testdict = base_override_dict.copy()
        testdict['calibration']['linkcal']['include'] = 'fiberflatnight'
        proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
        for job in ['linkcal', 'ccdcalib', 'psfnight']:
            self.assertTrue(job in proctable['JOBDESC'])
        for job in ['nightlybias', 'nightlyflat']:
            self.assertTrue(job not in proctable['JOBDESC'])

        ## Test link biasnight
        testdict = base_override_dict.copy()
        testdict['calibration']['linkcal']['include'] = 'biasnight'
        proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
        for job in ['linkcal', 'ccdcalib', 'psfnight', 'nightlyflat']:
            self.assertTrue(job in proctable['JOBDESC'])
        for job in ['nightlybias']:
            self.assertTrue(job not in proctable['JOBDESC'])

        ## Test link badcolumns
        testdict = base_override_dict.copy()
        testdict['calibration']['linkcal']['include'] = 'badcolumns'
        proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
        for job in ['linkcal', 'ccdcalib', 'psfnight', 'nightlyflat']:
            self.assertTrue(job in proctable['JOBDESC'])
        for job in ['nightlybias']:
            self.assertTrue(job not in proctable['JOBDESC'])

        ## Test link ctecorrnight
        testdict = base_override_dict.copy()
        testdict['calibration']['linkcal']['include'] = 'ctecorrnight'
        proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
        for job in ['linkcal', 'ccdcalib', 'psfnight', 'nightlyflat']:
            self.assertTrue(job in proctable['JOBDESC'])
        for job in ['nightlybias']:
            self.assertTrue(job not in proctable['JOBDESC'])

        ## Test link ctecorrnight and biasnight
        testdict = base_override_dict.copy()
        testdict['calibration']['linkcal']['include'] = 'ctecorrnight,biasnight'
        proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
        for job in ['linkcal', 'ccdcalib', 'psfnight', 'nightlyflat']:
            self.assertTrue(job in proctable['JOBDESC'])
        for job in ['nightlybias']:
            self.assertTrue(job not in proctable['JOBDESC'])

        ## Test link badcolumns and biasnight
        testdict = base_override_dict.copy()
        testdict['calibration']['linkcal']['include'] = 'badcolumns,biasnight'
        proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
        for job in ['linkcal', 'ccdcalib', 'psfnight', 'nightlyflat']:
            self.assertTrue(job in proctable['JOBDESC'])
        for job in ['nightlybias']:
            self.assertTrue(job not in proctable['JOBDESC'])

        ## Test link cte and badcol
        testdict = base_override_dict.copy()
        testdict['calibration']['linkcal']['include'] = 'ctecorrnight,badcolumns'
        proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
        for job in ['linkcal', 'nightlybias', 'psfnight', 'nightlyflat']:
            self.assertTrue(job in proctable['JOBDESC'])
        for job in ['ccdcalib']:
            self.assertTrue(job not in proctable['JOBDESC'])

        ## Test link ccdcalib
        calib_files = 'biasnight,badcolumns,ctecorrnight'
        testdict = base_override_dict.copy()
        testdict['calibration']['linkcal']['include'] = calib_files
        proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
        for job in ['linkcal', 'psfnight', 'nightlyflat']:
            self.assertTrue(job in proctable['JOBDESC'])
        for job in ['nightlybias', 'ccdcalib']:
            self.assertTrue(job not in proctable['JOBDESC'])

        ## Test link psfnight and fiberflatnight
        testdict = base_override_dict.copy()
        testdict['calibration']['linkcal']['include'] = 'psfnight,fiberflatnight'
        proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
        for job in ['linkcal', 'ccdcalib']:
            self.assertTrue(job in proctable['JOBDESC'])
        for job in ['nightlybias', 'psfnight', 'nightlyflat']:
            self.assertTrue(job not in proctable['JOBDESC'])

        ## Test link everything except fiberflatnight -- should raise error
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
        calib_files = 'biasnight,badcolumns,ctecorrnight,psfnight'
        testdict = base_override_dict.copy()
        testdict['calibration']['linkcal']['include'] = calib_files
        proctable, unproctable = self._override_write_run_delete(testdict,
                                                                 dry_run_level=3,
                                                                 psf_linking_without_fflat=True)
        for job in ['linkcal', 'nightlyflat']:
            self.assertTrue(job in proctable['JOBDESC'])
        for job in ['nightlybias', 'ccdcalib', 'psfnight']:
            self.assertTrue(job not in proctable['JOBDESC'])

        ## Test link everything except psfnight
        calib_files = 'biasnight,badcolumns,ctecorrnight,fiberflatnight'
        testdict = base_override_dict.copy()
        testdict['calibration']['linkcal']['include'] = calib_files
        proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
        for job in ['linkcal', 'psfnight']:
            self.assertTrue(job in proctable['JOBDESC'])
        for job in ['nightlybias', 'ccdcalib', 'nightlyflat']:
            self.assertTrue(job not in proctable['JOBDESC'])

        ## Test linking an earlier night without 1s CTE flat
        testdict = base_override_dict.copy()
        testnight = self.basicnight
        testdict['calibration']['linkcal']['refnight'] = testnight-1
        testdict['calibration']['linkcal']['include'] = 'biasnight,badcolumns'
        proctable, unproctable = self._override_write_run_delete(testdict, night=testnight, dry_run_level=3)
        for job in ['linkcal', 'psfnight', 'nightlyflat', 'tilenight']:
            self.assertIn(job, set(proctable['JOBDESC']))
        for job in ['nightlybias', 'ccdcalib']:
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
            self.assertTrue(job in proctable['JOBDESC'])
        for job in ['linkcal', 'nightlybias']:
            self.assertTrue(job not in proctable['JOBDESC'])
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
            self.assertTrue(job in proctable['JOBDESC'])
        for job in ['linkcal', 'nightlybias']:
            self.assertTrue(job not in proctable['JOBDESC'])
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
                if has_1secflat:
                    ## if 1sec flat has arrived, cals should be submitted.
                    ## Note: this could be different if we switch to testing a daily night with
                    ## and override file, in which case e.g. it could have linkcal instead of nightlyflat
                    for jobdesc in ('ccdcalib', 'arc', 'psfnight', 'flat', 'nightlyflat'):
                        self.assertIn(jobdesc, proctable['JOBDESC'])
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

