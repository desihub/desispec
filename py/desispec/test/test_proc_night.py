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

from desispec.workflow.tableio import load_table
from desispec.workflow.redshifts import get_ztile_script_pathname
from desispec.workflow.desi_proc_funcs import get_desi_proc_tilenight_batch_file_pathname
from desispec.io import findfile

from desispec.scripts.proc_night import proc_night
from desiutil.log import get_logger

class TestProcNight(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.reduxdir = tempfile.mkdtemp()
        cls.specprod = 'test'
        cls.proddir = os.path.join(cls.reduxdir, cls.specprod)
        cls.night = 20230914

        cls.origenv = os.environ.copy()
        os.environ['DESI_SPECTRO_REDUX'] = cls.reduxdir
        os.environ['SPECPROD'] = cls.specprod
        os.environ['NERSC_HOST'] = 'perlmutter'  # pretend to be on Perlmutter for testing
        os.environ['DESI_LOGLEVEL'] = 'WARNING' # reduce output from all the proc_night calls

        os.makedirs(cls.proddir)
        expdir = importlib.resources.files('desispec').joinpath('test', 'data', 'exposure_tables')
        shutil.copytree(expdir, os.path.join(cls.proddir, 'exposure_tables'))

        cls.etable_file = findfile('exposure_table', cls.night)
        cls.etable = load_table(cls.etable_file)
        cls.override_file = findfile('override', cls.night) # these are created in function


    def tearDown(self):
        # remove everything from prod except exposure_tables
        for path in glob.glob(self.proddir+'/*'):
            if os.path.basename(path) == 'exposure_tables':
                pass
            elif os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        # remove override_file if leftover from failed test
        if os.path.isfile(self.override_file):
            os.remove(self.override_file)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.reduxdir)
        for key in ('DESI_SPECTRO_REDUX', 'SPECPROD', 'NERSC_HOST', 'DESI_LOGLEVEL'):
            if key in cls.origenv:
                os.environ[key] = cls.origenv[key]
            else:
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

    def _override_write_run_delete(self, override_dict, **kwargs):
        """Write override, run proc_night, remove override file, and return outputs"""
        with open(self.override_file, 'w') as fil:
            yaml.safe_dump(override_dict, fil)
        proctable, unproctable = proc_night(self.night, sub_wait_time=0.0, **kwargs)
        os.remove(self.override_file)
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

        # ## Test no psfnight but still fiberflatnight -- should raise error
        # testdict = base_override_dict.copy()
        # testdict['calibration']['linkcal']['include'] = 'psfnight'
        # proctable, unproctable = self._override_write_run_delete(testdict)

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

        # ## Test link everything except fiberflatnight -- should raise error
        # calib_files = 'biasnight,badcolumns,ctecorrnight,psfnight'
        # testdict = base_override_dict.copy()
        # testdict['calibration']['linkcal']['include'] = calib_files
        # proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)

        ## Test link everything except psfnight
        calib_files = 'biasnight,badcolumns,ctecorrnight,fiberflatnight'
        testdict = base_override_dict.copy()
        testdict['calibration']['linkcal']['include'] = calib_files
        proctable, unproctable = self._override_write_run_delete(testdict, dry_run_level=3)
        for job in ['linkcal', 'psfnight']:
            self.assertTrue(job in proctable['JOBDESC'])
        for job in ['nightlybias', 'ccdcalib', 'nightlyflat']:
            self.assertTrue(job not in proctable['JOBDESC'])