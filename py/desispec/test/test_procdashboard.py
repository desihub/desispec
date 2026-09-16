# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.scripts.procdashboard
"""

import os
import importlib
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from desispec.io import findfile
from desispec.io.util import decode_camword
from desispec.scripts.procdashboard import DASHBOARD_COLNAMES, \
    populate_exp_night_info
from desispec.workflow.proc_dashboard_funcs import generate_nightly_table_html
from desispec.workflow.proctable import default_prow, \
    instantiate_processing_table
from desispec.workflow.tableio import write_table

## night of the canned exposure table used here, which has zeros, darks, arcs,
## flats, and science exposures, all with camword a0123456789
_night = 20250318
_camword = 'a0123456789'
_ncams = len(decode_camword(_camword))


class TestProcDashboard(unittest.TestCase):
    """Test that the dashboard reports the nightly calibration jobs"""

    @classmethod
    def setUpClass(cls):
        cls.night = _night
        cls.reduxdir = tempfile.mkdtemp()
        cls.specprod = 'test'
        cls.proddir = os.path.join(cls.reduxdir, cls.specprod)

        cls.origenv = os.environ.copy()
        os.environ['DESI_SPECTRO_REDUX'] = cls.reduxdir
        os.environ['SPECPROD'] = cls.specprod
        os.environ['DESI_SPECTRO_DATA'] = os.path.join(cls.reduxdir, 'rawdata')
        os.environ['DESI_DASHBOARD'] = os.path.join(cls.proddir, 'run', 'dashboard')

        os.makedirs(os.environ['DESI_DASHBOARD'])
        expdir = importlib.resources.files('desispec').joinpath('test', 'data',
                                                                'exposure_tables')
        shutil.copytree(expdir, os.path.join(cls.proddir, 'exposure_tables'))

        etable = importlib.import_module('desispec.workflow.tableio').load_table(
                findfile('exposure_table', cls.night), tabletype='exptable')
        obstypes = np.array([str(o) for o in etable['OBSTYPE']])
        expids = np.array(etable['EXPID'], dtype=int)
        cls.zeros = expids[obstypes == 'zero']
        cls.darks = expids[obstypes == 'dark']
        cls.arcs = expids[obstypes == 'arc']

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.reduxdir, ignore_errors=True)
        for key in ['DESI_SPECTRO_REDUX', 'SPECPROD', 'DESI_SPECTRO_DATA',
                    'DESI_DASHBOARD']:
            if key in cls.origenv:
                os.environ[key] = cls.origenv[key]
            else:
                del os.environ[key]

    def tearDown(self):
        for subdir in ['calibnight', 'processing_tables', 'run/scripts']:
            shutil.rmtree(os.path.join(self.proddir, subdir), ignore_errors=True)

    #
    # Helpers
    #
    def _prow(self, jobdesc, expids, status='COMPLETED', obstype=None, intid=1):
        """Create a processing table row for a calibration job"""
        prow = default_prow()
        prow['EXPID'] = np.array(expids, dtype=int)
        prow['JOBDESC'] = jobdesc
        prow['OBSTYPE'] = obstype if obstype is not None else jobdesc
        prow['NIGHT'] = self.night
        prow['PROCCAMWORD'] = _camword
        prow['CALIBRATOR'] = 1
        prow['INTID'] = intid
        ## a non-positive QID keeps update_from_queue from reaching for Slurm,
        ## which the mock in _run_dashboard also guards against
        prow['LATEST_QID'] = 0
        prow['STATUS'] = status
        return prow

    def _write_proctable(self, prows):
        pathname = findfile('processing_table', night=self.night)
        os.makedirs(os.path.dirname(pathname), exist_ok=True)
        ptable = instantiate_processing_table(rows=prows)
        write_table(ptable, tablename=pathname, tabletype='proctable')

    def _touch_calibnight(self, prefix, ext='fits', cameras=None):
        """Create per-camera calibnight files, returning how many were made"""
        if cameras is None:
            cameras = decode_camword(_camword)
        caldir = os.path.join(self.proddir, 'calibnight', str(self.night))
        os.makedirs(caldir, exist_ok=True)
        for camera in cameras:
            fname = f'{prefix}-{camera}-{self.night}.{ext}'
            open(os.path.join(caldir, fname), 'w').close()
        return len(cameras)

    def _touch_log(self, basename, qid):
        """Create a batch script and its log, returning the log pathname"""
        logdir = os.path.join(self.proddir, 'run', 'scripts', 'night',
                              str(self.night))
        os.makedirs(logdir, exist_ok=True)
        logname = os.path.join(logdir, f'{basename}-{qid}.log')
        open(logname, 'w').close()
        open(os.path.join(logdir, f'{basename}.slurm'), 'w').close()
        return logname

    def _link_biasnight(self, cameras, ext='fits.gz'):
        """Create relative bias links to files on a reference night."""
        refnight = self.night - 1
        refdir = os.path.join(self.proddir, 'calibnight', str(refnight))
        caldir = os.path.join(self.proddir, 'calibnight', str(self.night))
        os.makedirs(refdir, exist_ok=True)
        os.makedirs(caldir, exist_ok=True)
        for camera in cameras:
            reffile = os.path.join(refdir, f'biasnight-{camera}-{refnight}.{ext}')
            newfile = os.path.join(caldir, f'biasnight-{camera}-{self.night}.{ext}')
            open(reffile, 'w').close()
            os.symlink(os.path.relpath(reffile, caldir), newfile)

    def _run_dashboard(self, **kwargs):
        """Run populate_exp_night_info without querying Slurm"""
        with patch('desispec.scripts.procdashboard.update_from_queue',
                   side_effect=lambda ptable, **kw: ptable) as mock:
            output = populate_exp_night_info(self.night, **kwargs)
        self.assertLessEqual(mock.call_count, 1)
        return output

    #
    # Tests
    #
    def test_calib_jobs_have_rows(self):
        """Every nightly calibration job should get a row of its own"""
        self._write_proctable([
            self._prow('linkcal', [], obstype='link', intid=1),
            self._prow('biaspdark', self.darks[:2], obstype='dark', intid=2),
            self._prow('pdark', self.darks[2:], obstype='dark', intid=3),
            self._prow('ccdcalib', [self.darks[0]], obstype='dark', intid=4),
            self._prow('psfnight', self.arcs, obstype='arc', intid=5),
            ])
        self._touch_calibnight('biasnight', ext='fits.gz')
        self._touch_calibnight('psfnight')

        output = self._run_dashboard()

        jobdescs = set(key.split('_')[0] for key in output)
        for jobdesc in ['linkcal', 'biaspdark', 'pdark', 'ccdcalib', 'psfnight']:
            self.assertIn(jobdesc, jobdescs, f'no dashboard row for {jobdesc}')

        ## the bias written by the biaspdark is complete, so is the psfnight
        biaspdark = self._get_row(output, 'biaspdark')
        self.assertEqual(biaspdark['BIAS'], f'{_ncams}/{_ncams}')
        self.assertEqual(biaspdark['COLOR'], 'GOOD')
        self.assertEqual(self._get_row(output, 'psfnight')['COLOR'], 'GOOD')

    def test_biasnight_counts_cameras(self):
        """A bias missing some cameras should read as incomplete"""
        self._write_proctable([self._prow('biasnight', self.zeros[:1],
                                          obstype='zero')])
        cameras = decode_camword(_camword)[:4]
        self._touch_calibnight('biasnight', ext='fits.gz', cameras=cameras)

        row = self._get_row(self._run_dashboard(), 'biasnight')
        self.assertEqual(row['BIAS'], f'{len(cameras)}/{_ncams}')
        self.assertEqual(row['COLOR'], 'INCOMPLETE')

    def test_biasnight_with_no_files_is_bad(self):
        """A bias job that wrote nothing should be flagged, not left gray"""
        self._write_proctable([self._prow('biasnight', self.zeros[:1],
                                          obstype='zero')])

        row = self._get_row(self._run_dashboard(), 'biasnight')
        self.assertEqual(row['BIAS'], f'0/{_ncams}')
        self.assertEqual(row['COLOR'], 'BAD')

    def test_status_only_jobs_follow_the_queue(self):
        """A linkcal without biases, ccdcalib and pdark follow Slurm status."""
        for status, color in [('COMPLETED', 'GOOD'), ('FAILED', 'BAD'),
                              ('TIMEOUT', 'BAD'), ('RUNNING', 'RUNNING'),
                              ('PENDING', 'PENDING'), ('UNSUBMITTED', 'BAD'),
                              ('DEP_NOT_SUBD', 'BAD'), ('MAX_RESUB', 'BAD'),
                              ('UNKNOWN', 'INCOMPLETE')]:
            with self.subTest(status=status):
                self._write_proctable([
                    self._prow('linkcal', [], status=status, obstype='link',
                               intid=1),
                    self._prow('ccdcalib', [self.darks[0]], status=status,
                               obstype='dark', intid=2),
                    self._prow('pdark', self.darks, status=status,
                               obstype='dark', intid=3),
                    ])
                with patch('desispec.scripts.procdashboard.load_override_file',
                           return_value={'calibration': {'linkcal': {'exclude': 'biasnight'}}}):
                    output = self._run_dashboard()
                for jobdesc in ['linkcal', 'ccdcalib', 'pdark']:
                    row = self._get_row(output, jobdesc)
                    self.assertEqual(row['COLOR'], color,
                                     f'{jobdesc} with status {status}')
                    self.assertEqual(row['STATUS'], status)
                    ## No bias links are expected when biases are excluded.
                    expected_bias = '0/0' if jobdesc == 'linkcal' else '----'
                    self.assertEqual(row['BIAS'], expected_bias)
                    self.assertEqual(row['PSF'], '----')
                if color == 'BAD':
                    calib_rows = {key: dict(row) for key, row in output.items()
                                  if row['OBSTYPE'] in ['linkcal', 'ccdcalib', 'pdark']}
                    html, night_status = generate_nightly_table_html(calib_rows, self.night, show_null=True)
                    self.assertEqual(night_status, 'BAD')
                self.tearDown()

    def test_status_only_cached_rows_follow_current_status(self):
        """Old NULL rows and previously completed jobs must not freeze status."""
        for cached_color in ['NULL', 'GOOD']:
            for status, color in [('FAILED', 'BAD'), ('RUNNING', 'RUNNING'), ('COMPLETED', 'GOOD')]:
                with self.subTest(cached_color=cached_color, status=status):
                    prows = [self._prow('linkcal', [], obstype='link', intid=1),
                             self._prow('ccdcalib', self.darks[:1], intid=2),
                             self._prow('pdark', self.darks, intid=3)]
                    self._write_proctable(prows)
                    with patch('desispec.scripts.procdashboard.load_override_file',
                               return_value={'calibration': {'linkcal': {'exclude': 'biasnight'}}}):
                        cached = self._run_dashboard()
                        for prow in prows:
                            row = self._get_row(cached, prow['JOBDESC'])
                            row['COLOR'] = cached_color
                            row['STATUS'] = 'MAX_RESUB' if cached_color == 'NULL' else 'COMPLETED'
                            prow['STATUS'] = status
                        self._write_proctable(prows)
                        output = self._run_dashboard(night_json_info=cached)
                    for prow in prows:
                        row = self._get_row(output, prow['JOBDESC'])
                        self.assertEqual(row['STATUS'], status)
                        self.assertEqual(row['COLOR'], color)
                    self.tearDown()

    def test_bias_compression_formats(self):
        """Count compressed and uncompressed biases while separating links."""
        for jobdesc in ['biasnight', 'biaspdark']:
            for file_ext, link_ext in [('fits', 'fits'), ('fits', 'fits.gz'), ('fits.gz', 'fits')]:
                with self.subTest(jobdesc=jobdesc, file_ext=file_ext, link_ext=link_ext):
                    expids = self.zeros[:1] if jobdesc == 'biasnight' else self.darks
                    bias = self._prow(jobdesc, expids)
                    bias['PROCCAMWORD'] = 'a1'
                    link = self._prow('linkcal', [], obstype='link', intid=2)
                    link['PROCCAMWORD'] = 'a0'
                    self._write_proctable([bias, link])
                    self._touch_calibnight('biasnight', ext=file_ext, cameras=decode_camword('a1'))
                    self._link_biasnight(decode_camword('a0'), ext=link_ext)
                    self._touch_calibnight('biasnight', ext='fits.bak', cameras=['b1'])

                    ## Detect either format even when it differs from the current
                    ## writing preference, as it can for linked reference files.
                    with patch.dict(os.environ, {'DESI_COMPRESSION': 'NONE'}):
                        output = self._run_dashboard()
                    for desc in [jobdesc, 'linkcal']:
                        row = self._get_row(output, desc)
                        self.assertEqual(row['BIAS'], '3/3')
                        self.assertEqual(row['COLOR'], 'GOOD')
                    self.tearDown()

    def test_bias_files_and_links_are_counted_separately(self):
        """Linked biases must not inflate counts for bias-producing jobs."""
        for jobdesc in ['biasnight', 'biaspdark']:
            with self.subTest(jobdesc=jobdesc):
                expids = self.zeros[:1] if jobdesc == 'biasnight' else self.darks
                bias = self._prow(jobdesc, expids)
                bias['PROCCAMWORD'] = 'a123456789'
                link = self._prow('linkcal', [], obstype='link', intid=2)
                link['PROCCAMWORD'] = 'a0'
                self._write_proctable([bias, link])
                self._touch_calibnight('biasnight', ext='fits.gz', cameras=decode_camword('a123456789'))
                self._link_biasnight(decode_camword('a0'))

                output = self._run_dashboard()
                row = self._get_row(output, jobdesc)
                self.assertEqual(row['BIAS'], '27/27')
                self.assertEqual(row['COLOR'], 'GOOD')
                self.assertEqual(self._get_row(output, 'linkcal')['BIAS'], '3/3')
                self.tearDown()

    def test_extra_bias_files_and_links_remain_visible(self):
        """Do not filter either count to the job's expected camera set."""
        bias = self._prow('biasnight', self.zeros[:1], obstype='zero')
        bias['PROCCAMWORD'] = 'a1'
        link = self._prow('linkcal', [], obstype='link', intid=2)
        link['PROCCAMWORD'] = 'a0'
        self._write_proctable([bias, link])
        self._touch_calibnight('biasnight', ext='fits.gz', cameras=decode_camword('a12'))
        self._link_biasnight(decode_camword('a03'))

        output = self._run_dashboard()
        row = self._get_row(output, 'biasnight')
        self.assertEqual(row['BIAS'], '6/3')
        self.assertEqual(row['COLOR'], 'OVERFULL')
        self.assertEqual(self._get_row(output, 'linkcal')['BIAS'], '6/3')
        self.assertEqual(self._get_row(output, 'linkcal')['COLOR'], 'OVERFULL')

    def test_linkcal_bias_counts_affect_color(self):
        """Missing links affect completed jobs without masking queue status."""
        for nlinks, status, color in [
                (0, 'COMPLETED', 'BAD'), (1, 'COMPLETED', 'INCOMPLETE'),
                (3, 'COMPLETED', 'GOOD'), (4, 'COMPLETED', 'OVERFULL'),
                (0, 'PENDING', 'PENDING'), (1, 'RUNNING', 'RUNNING'),
                (3, 'FAILED', 'BAD')]:
            with self.subTest(nlinks=nlinks, status=status):
                link = self._prow('linkcal', [], status=status, obstype='link')
                link['PROCCAMWORD'] = 'a0'
                self._write_proctable([link])
                self._link_biasnight(['b0', 'r0', 'z0', 'b1'][:nlinks])
                row = self._get_row(self._run_dashboard(), 'linkcal')
                self.assertEqual(row['BIAS'], f'{nlinks}/3')
                self.assertEqual(row['COLOR'], color)
                self.tearDown()

    def test_linkcal_bias_expectations(self):
        """Bias overrides can differ from the linkcal's other products."""
        self._write_proctable([self._prow('linkcal', [], obstype='link')])
        self._link_biasnight(decode_camword('a0'))
        for settings, expected in [
                ({'include': 'biasnight,psfnight', 'biaslink_camword': 'a0'}, '3/3'),
                ({'include': 'psfnight,fiberflatnight'}, '3/0'),
                ({'exclude': 'biasnight'}, '3/0'),
                ({'exclude': 'psfnight'}, f'3/{_ncams}'),
                ({}, f'3/{_ncams}')]:
            with self.subTest(settings=settings):
                with patch('desispec.scripts.procdashboard.load_override_file',
                           return_value={'calibration': {'linkcal': settings}}):
                    row = self._get_row(self._run_dashboard(), 'linkcal')
                self.assertEqual(row['BIAS'], expected)

    def test_cached_bias_counts_are_refreshed(self):
        """Cached rows must not retain counts that used to include links."""
        bias = self._prow('biasnight', self.zeros[:1], obstype='zero')
        bias['PROCCAMWORD'] = 'a0'
        link = self._prow('linkcal', [], obstype='link', intid=2)
        link['PROCCAMWORD'] = 'a1'
        self._write_proctable([bias, link])
        self._touch_calibnight('biasnight', ext='fits.gz', cameras=decode_camword('a0'))
        self._link_biasnight(decode_camword('a1'))
        cached = self._run_dashboard()
        self._get_row(cached, 'biasnight')['BIAS'] = '6/3'
        self._get_row(cached, 'linkcal')['BIAS'] = '----'

        output = self._run_dashboard(night_json_info=cached)
        self.assertEqual(self._get_row(output, 'biasnight')['BIAS'], '3/3')
        self.assertEqual(self._get_row(output, 'linkcal')['BIAS'], '3/3')

    def test_linkcal_row_without_exposures(self):
        """A linkcal has no exposure of the night, but still gets a row"""
        self._write_proctable([self._prow('linkcal', [], status='FAILED',
                                          obstype='link')])
        logname = self._touch_log(f'linkcal-{self.night}-{_camword}', 987654)

        row = self._get_row(self._run_dashboard(), 'linkcal')
        self.assertEqual(row['EXPID'], '----')
        self.assertEqual(row['COLOR'], 'BAD')
        self.assertIn(os.path.basename(logname), row['LOG FILE'])
        self.assertIn('.slurm', row['SLURM FILE'])

    def test_linkcal_comment_describes_the_link(self):
        """The exposures of a linkcal are the reference night's, so say what
        was linked instead"""
        override = findfile('override', night=self.night)
        os.makedirs(os.path.dirname(override), exist_ok=True)
        with open(override, 'w') as fil:
            fil.write('calibration:\n  linkcal:\n'
                      + '    include: biasnight,psfnight\n'
                      + '    refnight: 20250317\n')
        self.addCleanup(os.remove, override)

        ## the exposures a linkcal carries belong to the reference night
        self._write_proctable([self._prow('linkcal', [123456],
                                          obstype='link')])

        comment = self._get_row(self._run_dashboard(), 'linkcal')['COMMENTS']
        self.assertIn('biasnight,psfnight', comment)
        self.assertIn('20250317', comment)
        self.assertNotIn('123456', comment)

    def test_log_links_use_the_first_exposure(self):
        """Calib job scripts are named for the first exposure they processed"""
        self._write_proctable([self._prow('biaspdark', self.darks,
                                          status='FAILED', obstype='dark')])
        zexpid = str(self.darks[0]).zfill(8)
        self._touch_log(f'biaspdark-{self.night}-{zexpid}-{_camword}', 11111)
        newest = self._touch_log(f'biaspdark-{self.night}-{zexpid}-{_camword}',
                                 22222)

        row = self._get_row(self._run_dashboard(), 'biaspdark')
        ## the most recent submission is the one worth looking at
        self.assertIn(os.path.basename(newest), row['LOG FILE'])
        self.assertIn(f'biaspdark-{self.night}-{zexpid}-{_camword}.slurm',
                      row['SLURM FILE'])

    def test_duplicate_calib_rows_are_collapsed(self):
        """Repeated rows for one job shouldn't become repeated dashboard rows"""
        prows = [self._prow('biasnight', self.zeros[:1], obstype='zero',
                            intid=i) for i in range(5)]
        self._write_proctable(prows)
        self._touch_calibnight('biasnight', ext='fits.gz')

        output = self._run_dashboard()
        biasrows = [key for key in output if key.startswith('biasnight')]
        self.assertEqual(len(biasrows), 1)

    def test_all_rows_share_the_same_columns(self):
        """Mismatched keys would misalign the columns of the html table"""
        self._write_proctable([
            self._prow('linkcal', [], obstype='link', intid=1),
            self._prow('biaspdark', self.darks, obstype='dark', intid=2),
            self._prow('psfnight', self.arcs, obstype='arc', intid=3),
            ])
        self._touch_calibnight('biasnight', ext='fits.gz')

        output = self._run_dashboard()
        self.assertGreater(len(output), 0)
        for key, row in output.items():
            self.assertEqual(list(row.keys()), DASHBOARD_COLNAMES, key)

        ## and the rendered table has as many headers as it has cells per row
        html, status = generate_nightly_table_html(output, self.night,
                                                   show_null=True)
        nheaders = html.count('<th>')
        self.assertEqual(nheaders, len(DASHBOARD_COLNAMES) - 1)  # COLOR isn't shown
        datarows = [line for line in html.split('\n') if '<td' in line]
        self.assertEqual(len(datarows), len(output))
        for line in datarows:
            self.assertEqual(line.count('<td'), nheaders, line[:80])

    def test_stale_cache_entries_are_regenerated(self):
        """A row cached before a column existed must not be reused"""
        self._write_proctable([self._prow('biasnight', self.zeros[:1],
                                          obstype='zero')])
        self._touch_calibnight('biasnight', ext='fits.gz')

        fresh = self._run_dashboard()
        key = [k for k in fresh if k.startswith('biasnight')][0]

        stale = {k: dict(v) for k, v in fresh.items()}
        stale[key].pop('BIAS')
        stale[key]['COLOR'] = 'GOOD'

        output = self._run_dashboard(night_json_info=stale)
        self.assertEqual(list(output[key].keys()), DASHBOARD_COLNAMES)
        self.assertEqual(output[key]['BIAS'], f'{_ncams}/{_ncams}')

    def _get_row(self, output, jobdesc):
        """Return the single dashboard row for the given job description"""
        rows = [row for key, row in output.items()
                if key.split('_')[0] == jobdesc]
        self.assertEqual(len(rows), 1, f'expected one {jobdesc} row')
        return rows[0]
