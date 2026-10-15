"""Regression tests using a small synthetic production and real FITS I/O.

The test_tsnr_afterburner module covers isolated helpers. These tests cover
source fallback, incremental output updates, MPI gathering, and legacy CLI
behavior. Only calibration calculations and external GFA data are mocked.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack

from desispec.io import read_table
from desispec.scripts import tsnr_afterburner as mod
from desispec.test.create_tsnr_fixtures import (
    write_cframe_fixture, write_exptable_fixture, write_qa_fixture,
)


class TestAfterburnerIntegration(unittest.TestCase):
    """Exercise the public entry point against real workflow and FITS files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.prod = Path(self.tmp.name) / 'prod'
        self.prod.mkdir()
        self.outfile = self.prod / 'summary.fits'
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {
            'DESI_SPECTRO_REDUX': self.tmp.name, 'SPECPROD': 'prod',
            'DESI_SPECTRO_DATA': self.tmp.name + '/raw',
            'DESI_SURVEYOPS': self.tmp.name + '/ops',
        }))
        self.stack.enter_context(patch.object(mod, 'tsnr2_to_efftime', side_effect=lambda x, t: float(x) * .1))
        self.skymag = self.stack.enter_context(patch.object(mod, 'get_skymag_values', return_value={
            'SKY_MAG_G_SPEC': 22., 'SKY_MAG_R_SPEC': 21., 'SKY_MAG_Z_SPEC': 20.,
        }))
        self._add_exposure()

    def _add_exposure(self, night=20211001, expid=100, tileid=1234, laststep='all'):
        write_exptable_fixture(self.prod, night, expid, tileid=tileid, laststep=laststep)
        if laststep == 'all':
            write_qa_fixture(self.prod, night, expid, tileid=tileid)

    def _run(self, *options):
        return mod.main(['--prod', str(self.prod), '-o', str(self.outfile), *options])

    def _tables(self):
        return read_table(str(self.outfile), 'EXPOSURES'), read_table(str(self.outfile), 'FRAMES')

    @contextmanager
    def _calculation(self, band='B', alpha_only=False):
        results = {} if alpha_only else {
            'TSNR2_' + tracer + '_' + band: np.array([50., 50.]) for tracer in mod._TSNR2_TRACERS
        }
        with ExitStack() as stack:
            for name in ('read_frame', 'read_fiberflat', 'read_sky', 'read_flux_calibration'):
                stack.enter_context(patch.object(mod, name, return_value=MagicMock()))
            calc = stack.enter_context(patch.object(mod, 'calc_tsnr2', return_value=(results, 1.5)))
            yield calc

    def _mpi(self, extra_tables=()):
        comm = MagicMock()
        comm.Get_rank.return_value = 0
        comm.Get_size.return_value = 1 + len(extra_tables)
        comm.bcast.side_effect = lambda value, root: value
        comm.gather.side_effect = lambda value, root: [value, *[(frm, exp, None, []) for frm, exp in extra_tables]]
        self.stack.enter_context(patch('desispec.parallel.use_mpi', return_value=True))
        self.stack.enter_context(patch.dict(sys.modules, {
            'mpi4py': SimpleNamespace(MPI=SimpleNamespace(COMM_WORLD=comm)),
        }))
        return comm

    def _tile_capture(self):
        def compute(exposures, *args, **kwargs):
            tiles = np.unique(exposures['TILEID'])
            return Table({'TILEID': tiles, 'GOALTIME': np.full(len(tiles), 1000.)})
        return patch.object(mod, 'compute_tile_completeness_table', side_effect=compute)

    def test_executable_help(self):
        script = Path(__file__).resolve().parents[3] / 'bin' / 'desi_tsnr_afterburner'
        # the subprocess must import this checkout's desispec even when it is
        # only on sys.path through pytest (as in CI), not installed
        env = dict(os.environ)
        pydir = str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH'] = pydir + os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else pydir
        result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--recompute-skymags', result.stdout)

    def test_output_dtypes_match_production(self):
        # production exposures files store these columns as float64; EXPTIME
        # is float64 in EXPOSURES but float32 in FRAMES
        float64 = {'EXPOSURES': ['TILERA', 'TILEDEC', 'MJD', 'EXPTIME', 'EFFTIME_SPEC', 'TSNR2_LYA',
                                 'LYA_EFFTIME_DARK', 'SKY_MAG_G_SPEC', 'SKY_MAG_R_SPEC', 'SKY_MAG_Z_SPEC'],
                   'FRAMES': ['TILERA', 'TILEDEC', 'MJD', 'TSNR2_LYA']}
        float32 = {'EXPOSURES': ['TSNR2_ELG', 'LRG_EFFTIME_DARK', 'EBV'],
                   'FRAMES': ['EXPTIME', 'TSNR2_ELG', 'EBV']}
        self._add_exposure(expid=101, laststep='skysub')
        with self._no_fiberassign():
            self.assertEqual(self._run('--add-badexp'), 0)
        exposures, frames = self._tables()
        for name, table in (('EXPOSURES', exposures), ('FRAMES', frames)):
            for col in float64[name]:
                self.assertEqual(table[col].dtype.kind + str(table[col].dtype.itemsize), 'f8', name + ' ' + col)
            for col in float32[name]:
                self.assertEqual(table[col].dtype.kind + str(table[col].dtype.itemsize), 'f4', name + ' ' + col)
        # header values keep full float64 precision
        self.assertEqual(float(exposures['TILERA'][exposures['EXPID'] == 100][0]), 180.)

    def test_empty_tables_dtypes_match_production(self):
        frames, exposures = mod._empty_tables()
        for col in ('TILERA', 'EXPTIME', 'EFFTIME_SPEC', 'TSNR2_LYA', 'SKY_MAG_R_SPEC', 'EFFTIME_GFA'):
            self.assertEqual(exposures[col].dtype.itemsize, 8, col)
        for col in ('TILERA', 'TSNR2_LYA'):
            self.assertEqual(frames[col].dtype.itemsize, 8, col)
        for col in ('EXPTIME', 'TSNR2_ELG'):
            self.assertEqual(frames[col].dtype.itemsize, 4, col)

    def test_frames_ebv_is_per_petal(self):
        # FRAMES gets each petal's median EBV; EXPOSURES the median over all fibers
        write_exptable_fixture(self.prod, 20211001, 100, camword='a01')
        write_qa_fixture(self.prod, 20211001, 100, fiber_ebv={0: [.01, .02, .03, 0.], 1: [.10, .11, .12]})
        petalqa = fits.getdata(mod.findfile('exposureqa', night=20211001, expid=100), 'PETALQA')
        table = Table(petalqa)
        table.add_row(table[0])
        table['PETAL_LOC'][1] = 1
        with fits.open(mod.findfile('exposureqa', night=20211001, expid=100), mode='update') as hdus:
            hdus['PETALQA'] = fits.BinTableHDU(table, name='PETALQA')
        self.assertEqual(self._run(), 0)
        exposures, frames = self._tables()
        self.assertAlmostEqual(float(exposures['EBV'][0]), .065, places=6)
        for camera, expected in (('b0', .02), ('z0', .02), ('r1', .11)):
            self.assertAlmostEqual(float(frames['EBV'][frames['CAMERA'] == camera][0]), expected, places=6)

    def test_frames_ebv_from_cframe_fibermap(self):
        # a camera read from its cframe keeps that cframe's fibermap EBV
        write_qa_fixture(self.prod, 20211001, 100, missing=['TSNR2_ELG_B'], fiber_ebv={0: [.01, .02, .03]})
        write_cframe_fixture(self.prod, 20211001, 100, 'b0')
        self.assertEqual(self._run(), 0)
        _, frames = self._tables()
        self.assertAlmostEqual(float(frames['EBV'][frames['CAMERA'] == 'b0'][0]), .05, places=6)
        self.assertAlmostEqual(float(frames['EBV'][frames['CAMERA'] == 'r0'][0]), .02, places=6)

    def test_frames_ebv_falls_back_to_exposure_value(self):
        # FIBERQA without PETAL_LOC (fixture default): every camera gets the exposure EBV
        self.assertEqual(self._run(), 0)
        exposures, frames = self._tables()
        np.testing.assert_allclose(frames['EBV'], exposures['EBV'][0])

    def test_night_without_science_keeps_string_widths(self):
        # a night with no science exposures merges an empty table; stored
        # string widths must not change
        self._run()
        before, before_frames = self._tables()
        write_exptable_fixture(self.prod, 20211002, 200, tileid=0)
        self.assertEqual(self._run('--nights', '20211002'), 0)
        after, after_frames = self._tables()
        for old, new in ((before, after), (before_frames, after_frames)):
            for col in old.colnames:
                if old[col].dtype.kind in 'US':
                    self.assertEqual(new[col].dtype, old[col].dtype, col)

    def test_qa_fast_path_needs_no_cframes(self):
        with patch.object(mod, 'read_one_camera') as camera:
            self.assertEqual(self._run(), 0)
        camera.assert_not_called()
        exposures, frames = self._tables()
        self.assertEqual(float(exposures['TSNR2_ELG'][0]), 30.)
        self.assertEqual(len(frames), 3)

    def test_single_zero_qa_value_is_not_missing(self):
        # e.g. TSNR2_LYA_Z is legitimately 0 in real QA files
        write_qa_fixture(self.prod, 20211001, 100, zeros=['TSNR2_LYA_Z'])
        with patch.object(mod, 'read_one_camera') as camera:
            self._run()
        camera.assert_not_called()
        exposures, _ = self._tables()
        self.assertEqual(float(exposures['TSNR2_LYA'][0]), 20.)
        self.assertEqual(float(exposures['TSNR2_ELG'][0]), 30.)

    def test_all_zero_qa_band_uses_scores(self):
        # exposure_qa leaves a band at 0 when it could not read that camera
        zeros = ['TSNR2_{}_B'.format(tracer) for tracer in mod._TSNR2_TRACERS]
        write_qa_fixture(self.prod, 20211001, 100, zeros=zeros)
        write_cframe_fixture(self.prod, 20211001, 100, 'b0')
        self._run()
        exposures, _ = self._tables()
        self.assertEqual(float(exposures['TSNR2_ELG'][0]), 40.)

    def test_old_qa_header_uses_exposure_table_and_cframe_fibermap(self):
        # QA files before 20260601 lack FAFLAVOR; a few lack MJD-OBS/EXPTIME/AIRMASS
        write_exptable_fixture(self.prod, 20211001, 100, EXPTIME=950., AIRMASS=1.3, **{'MJD-OBS': 59489.})
        write_qa_fixture(self.prod, 20211001, 100, drop_keys=['MJD-OBS', 'EXPTIME', 'AIRMASS', 'FAFLAVOR'])
        write_cframe_fixture(self.prod, 20211001, 100, 'b0')
        with patch.object(mod, 'read_one_camera') as camera:
            self.assertEqual(self._run(), 0)
        camera.assert_not_called()
        exposures, frames = self._tables()
        self.assertEqual(float(exposures['EXPTIME'][0]), 950.)
        self.assertAlmostEqual(float(exposures['AIRMASS'][0]), 1.3, places=5)
        self.assertEqual(float(exposures['MJD'][0]), 59489.)
        self.assertEqual(exposures['FAFLAVOR'][0], 'maindark')
        self.assertEqual(exposures['PROGRAM'][0], 'dark')
        self.assertEqual(set(frames['FAFLAVOR']), {'maindark'})

    def test_missing_qa_column_uses_scores_without_calibrations(self):
        write_qa_fixture(self.prod, 20211001, 100, missing=['TSNR2_ELG_B'])
        write_cframe_fixture(self.prod, 20211001, 100, 'b0')
        with patch.object(mod, 'calc_tsnr2') as calc, patch.object(mod, 'read_fiberflat') as flat:
            self._run()
        calc.assert_not_called()
        flat.assert_not_called()
        self.skymag.assert_not_called()
        exposures, _ = self._tables()
        self.assertEqual(float(exposures['TSNR2_ELG'][0]), 40.)
        self.assertEqual(float(exposures['TSNR2_LRG'][0]), 30.)

    def test_absent_petalqa_uses_scores(self):
        write_qa_fixture(self.prod, 20211001, 100, petal_hdu=False)
        for camera in ('b0', 'r0', 'z0'):
            write_cframe_fixture(self.prod, 20211001, 100, camera)
        with patch.object(mod, 'calc_tsnr2') as calc:
            self._run()
        calc.assert_not_called()
        self.assertEqual(float(self._tables()[0]['TSNR2_ELG'][0]), 60.)

    def test_absent_qa_file_uses_cframe_metadata(self):
        qa = write_qa_fixture(self.prod, 20211001, 100)
        Path(qa).unlink()
        write_cframe_fixture(self.prod, 20211001, 100, 'b0')
        self._run('--cameras', 'b0')
        exposures, _ = self._tables()
        self.assertEqual(int(exposures['TILEID'][0]), 1234)
        self.assertEqual(float(exposures['TSNR2_ELG'][0]), 20.)

    def test_absent_fiberqa_uses_cframe_metadata(self):
        qa = write_qa_fixture(self.prod, 20211001, 100)
        fits.HDUList([fits.PrimaryHDU()]).writeto(qa, overwrite=True)
        write_cframe_fixture(self.prod, 20211001, 100, 'b0')
        self._run('--cameras', 'b0')
        self.assertEqual(float(self._tables()[0]['TSNR2_ELG'][0]), 20.)
        self.skymag.assert_called_once()

    def test_missing_scores_calculates_only_the_needed_camera(self):
        write_qa_fixture(self.prod, 20211001, 100, missing=['TSNR2_ELG_B'])
        write_cframe_fixture(self.prod, 20211001, 100, 'b0', scores=False)
        with self._calculation() as calc:
            self._run()
        calc.assert_called_once()
        exposures, frames = self._tables()
        self.assertEqual(float(exposures['TSNR2_ELG'][0]), 70.)
        self.assertEqual(float(exposures['TSNR2_LRG'][0]), 30.)
        self.assertEqual(float(frames['TSNR2_ALPHA'][frames['CAMERA'] == 'b0'][0]), 1.5)

    def test_missing_score_column_preserves_stored_values(self):
        write_qa_fixture(self.prod, 20211001, 100, petal_hdu=False)
        write_cframe_fixture(self.prod, 20211001, 100, 'b0', missing=['TSNR2_ELG_B'])
        with self._calculation() as calc:
            self._run('--cameras', 'b0')
        calc.assert_called_once()
        exposures, _ = self._tables()
        self.assertEqual(float(exposures['TSNR2_ELG'][0]), 50.)
        self.assertEqual(float(exposures['TSNR2_LRG'][0]), 20.)

    def test_missing_calibration_leaves_existing_output_intact(self):
        self._run()
        before = self.outfile.read_bytes()
        write_qa_fixture(self.prod, 20211001, 100, missing=['TSNR2_ELG_B'])
        filename = write_cframe_fixture(self.prod, 20211001, 100, 'b0', scores=False)
        Path(fits.getheader(filename)['FIBERFLT']).unlink()
        with self.assertRaises(FileNotFoundError):
            self._run()
        self.assertEqual(self.outfile.read_bytes(), before)

    def test_recompute_overrides_qa_and_scores(self):
        write_cframe_fixture(self.prod, 20211001, 100, 'b0')
        with self._calculation() as calc:
            self._run('--recompute', '--cameras', 'b0')
        self.assertFalse(calc.call_args.kwargs['alpha_only'])
        self.assertEqual(float(self._tables()[0]['TSNR2_ELG'][0]), 50.)

    def test_recompute_ignores_details_cache(self):
        write_cframe_fixture(self.prod, 20211001, 100, 'b0')
        details = Path(self.tmp.name) / 'details'
        cache = details / '20211001' / '00000100' / 'tsnr-b0-00000100.fits'
        cache.parent.mkdir(parents=True)
        Table({'TSNR2_{}_B'.format(t): np.array([999.], dtype=np.float32)
               for t in mod._TSNR2_TRACERS}).write(str(cache))
        with self._calculation() as calc:
            self._run('--recompute', '--cameras', 'b0', '--details-dir', str(details))
        calc.assert_called_once()
        self.assertEqual(float(self._tables()[0]['TSNR2_ELG'][0]), 50.)
        self.assertEqual(float(np.median(Table.read(str(cache))['TSNR2_ELG_B'])), 50.)

    def test_alpha_only_preserves_qa_tsnr_and_writes_alpha(self):
        self._run()
        write_cframe_fixture(self.prod, 20211001, 100, 'b0', scores=False)
        with self._calculation(alpha_only=True) as calc:
            self._run('--alpha_only', '--cameras', 'b0')
        self.assertTrue(calc.call_args.kwargs['alpha_only'])
        exposures, frames = self._tables()
        self.assertEqual(float(exposures['TSNR2_ELG'][0]), 30.)
        self.assertEqual(float(frames['TSNR2_ALPHA'][frames['CAMERA'] == 'b0'][0]), 1.5)
        self._run('--cameras', 'b0')
        _, frames = self._tables()
        self.assertEqual(float(frames['TSNR2_ALPHA'][frames['CAMERA'] == 'b0'][0]), 1.5)

    def test_alpha_only_fills_missing_tsnr_when_required(self):
        write_qa_fixture(self.prod, 20211001, 100, missing=['TSNR2_ELG_B'])
        write_cframe_fixture(self.prod, 20211001, 100, 'b0', scores=False)
        with self._calculation() as calc:
            self._run('--alpha-only', '--cameras', 'b0')
        self.assertFalse(calc.call_args.kwargs['alpha_only'])
        exposures, _ = self._tables()
        self.assertEqual(float(exposures['TSNR2_ELG'][0]), 50.)
        self.assertEqual(float(exposures['TSNR2_LRG'][0]), 10.)

    def test_alpha_only_preserves_scores_without_qa(self):
        qa = write_qa_fixture(self.prod, 20211001, 100)
        Path(qa).unlink()
        write_cframe_fixture(self.prod, 20211001, 100, 'b0')
        with self._calculation(alpha_only=True) as calc:
            self._run('--alpha-only', '--cameras', 'b0')
        self.assertTrue(calc.call_args.kwargs['alpha_only'])
        exposures, frames = self._tables()
        self.assertEqual(float(exposures['TSNR2_ELG'][0]), 20.)
        self.assertEqual(float(frames['TSNR2_ALPHA'][0]), 1.5)

    def test_camera_update_reaggregates_retained_cameras(self):
        self._run()
        write_qa_fixture(self.prod, 20211001, 100, value=20.)
        self._run('--cameras', 'b0')
        exposures, frames = self._tables()
        self.assertEqual(float(exposures['TSNR2_ELG'][0]), 40.)
        self.assertEqual(float(np.sum(frames['TSNR2_ELG'])), 40.)

    def test_full_update_removes_newly_excluded_camera(self):
        self._run()
        write_exptable_fixture(self.prod, 20211001, 100, camword='b0')
        self._run()
        exposures, frames = self._tables()
        self.assertEqual(frames['CAMERA'].tolist(), ['b0'])
        self.assertEqual(float(exposures['TSNR2_ELG'][0]), 10.)

    def test_serial_and_mpi_reruns_are_identical(self):
        self._run()
        serial_exp, serial_frames = self._tables()
        self._mpi()
        self._run('--mpi')
        mpi_exp, mpi_frames = self._tables()
        # column by column so NaN (e.g. TSNR2_ALPHA from QA) compares equal
        for serial, mpi in ((serial_exp, mpi_exp), (serial_frames, mpi_frames)):
            self.assertEqual(serial.colnames, mpi.colnames)
            for col in serial.colnames:
                np.testing.assert_array_equal(serial[col], mpi[col])

    def test_mpi_gathers_other_rank_and_empty_rank(self):
        self._run()
        exposures, frames = self._tables()
        remote_exp, remote_frames = exposures.copy(), frames.copy()
        remote_exp['EXPID'] = 200
        remote_frames['EXPID'] = 200
        self._mpi([(remote_frames, remote_exp), mod._empty_tables()])
        self._run('--mpi')
        exposures, frames = self._tables()
        self.assertEqual(exposures['EXPID'].tolist(), [100, 200])
        self.assertEqual(len(frames), 6)

    def test_mpi_failure_preserves_existing_output(self):
        self._run()
        before = self.outfile.read_bytes()
        comm = self._mpi()
        with patch.object(mod, 'read_one_exposure', side_effect=RuntimeError('calculation failed')):
            self.assertEqual(self._run('--mpi'), 1)
        comm.gather.assert_called_once()
        self.assertEqual(self.outfile.read_bytes(), before)

    def test_remote_mpi_failure_preserves_existing_output(self):
        self._run()
        before = self.outfile.read_bytes()
        comm = self._mpi()
        frames, exposures = mod._empty_tables()
        comm.gather.side_effect = lambda value, root: [value, (frames, exposures, 'Rank 1: missing input', [])]
        self.assertEqual(self._run('--mpi'), 1)
        self.assertEqual(self.outfile.read_bytes(), before)

    def test_multiprocessing_dispatches_exposure_loader(self):
        with patch.object(mod.multiprocessing, 'Pool') as pool:
            pool.return_value.map.side_effect = lambda fn, args: list(map(fn, args))
            self.assertEqual(self._run('--nproc', '2'), 0)
        pool.assert_called_once_with(2)
        pool.return_value.close.assert_called_once()
        self.assertEqual(float(self._tables()[0]['TSNR2_ELG'][0]), 30.)

    def test_bad_exposure_skymags_use_the_pool(self):
        # one pool: exposure loading, then the bad-exposure sky magnitudes and fiberassign headers
        self._add_exposure(expid=101, laststep='skysub')
        os.makedirs(mod.os.path.dirname(mod.findfile('cframe', night=20211001, expid=101, camera='b0')))
        with patch.object(mod.multiprocessing, 'Pool') as pool, self._no_fiberassign():
            pool.return_value.map.side_effect = lambda fn, args: list(map(fn, args))
            self.assertEqual(self._run('--nproc', '2', '--add-badexp'), 0)
        pool.assert_called_once_with(2)
        functions = [call.args[0] for call in pool.return_value.map.call_args_list]
        self.assertEqual(functions, [mod._read_one_exposure_wrapper, mod._bad_exposure_skymags,
                                     mod._fiberassign_header])
        exposures, _ = self._tables()
        self.assertEqual(float(exposures['SKY_MAG_R_SPEC'][exposures['EXPID'] == 101][0]), 21.)

    def test_missing_fallback_camera_is_skipped(self):
        # b0 has neither a QA value nor a cframe: skip it, as the original afterburner did
        self._run()
        write_qa_fixture(self.prod, 20211001, 100, missing=['TSNR2_ELG_B'])
        self.assertEqual(self._run(), 0)
        exposures, frames = self._tables()
        self.assertEqual(frames['CAMERA'].tolist(), ['r0', 'z0'])
        self.assertEqual(float(exposures['TSNR2_ELG'][0]), 20.)

    def test_unreadable_exposure_gets_zeroed_row(self):
        # LASTSTEP=all exposure with no QA and no cframes: replace its old rows
        # with zeros (what is on disk), even without --add-badexp
        self._run()
        os.remove(mod.findfile('exposureqa', night=20211001, expid=100))
        self._add_exposure(expid=101)
        with self._no_fiberassign():
            self.assertEqual(self._run(), 0)
        exposures, frames = self._tables()
        self.assertEqual(exposures['EXPID'].tolist(), [100, 101])
        old = exposures[exposures['EXPID'] == 100]
        self.assertEqual(float(old['TSNR2_ELG'][0]), 0.)
        self.assertEqual(float(old['EFFTIME_SPEC'][0]), 0.)
        self.assertEqual(frames['CAMERA'][frames['EXPID'] == 100].tolist(), ['b0', 'r0', 'z0'])
        self.assertTrue(np.all(frames['TSNR2_ELG'][frames['EXPID'] == 100] == 0.))

    def test_missing_exposure_dir_skips_camera_checks(self):
        # no exposure directory: one check, no per-camera cframe lookups
        self._add_exposure(expid=101)
        expdir = os.path.dirname(mod.findfile('exposureqa', night=20211001, expid=101))
        os.remove(os.path.join(expdir, os.path.basename(mod.findfile('exposureqa', night=20211001, expid=101))))
        os.rmdir(expdir)
        with patch.object(mod, 'read_one_camera') as camera:
            self.assertIsNone(mod.read_one_exposure(20211001, 101))
        camera.assert_not_called()
        with self._no_fiberassign():
            self.assertEqual(self._run(), 0)
        exposures, frames = self._tables()
        self.assertEqual(float(exposures['EFFTIME_SPEC'][exposures['EXPID'] == 101][0]), 0.)
        self.assertEqual(int(np.sum(frames['EXPID'] == 101)), 3)

    def test_not_yet_processed_night_is_skipped(self):
        # a night with a processing table and expected exposures but no
        # exposures directory: nothing is read or written, with or without
        # --add-badexp, and existing rows are kept
        self._run()
        before = self._tables()[0]
        write_exptable_fixture(self.prod, 20211002, 200)
        write_exptable_fixture(self.prod, 20211002, 201, laststep='skysub')
        ptab = Path(mod.findfile('processing_table', night=20211002))
        ptab.parent.mkdir(parents=True, exist_ok=True)
        ptab.touch()
        with patch.object(mod, 'read_one_exposure', wraps=mod.read_one_exposure) as reader, self._no_fiberassign():
            self.assertEqual(self._run('--add-badexp'), 0)
        self.assertNotIn(20211002, [call.args[0] for call in reader.call_args_list])
        exposures, frames = self._tables()
        self.assertEqual(exposures['EXPID'].tolist(), before['EXPID'].tolist())
        self.assertNotIn(200, frames['EXPID'])

    def test_bad_only_night_without_exposures_dir(self):
        # only bad exposures and no exposures directory: added with --add-badexp
        write_exptable_fixture(self.prod, 20211002, 201, laststep='skysub')
        ptab = Path(mod.findfile('processing_table', night=20211002))
        ptab.parent.mkdir(parents=True, exist_ok=True)
        ptab.touch()
        with self._no_fiberassign():
            self.assertEqual(self._run('--add-badexp'), 0)
        exposures, _ = self._tables()
        self.assertIn(201, exposures['EXPID'])

    def test_night_without_science_contributes_nothing(self):
        # e.g. a darknight-only processing table: no science exposures
        write_exptable_fixture(self.prod, 20211002, 202, tileid=0)
        ptab = Path(mod.findfile('processing_table', night=20211002))
        ptab.parent.mkdir(parents=True, exist_ok=True)
        ptab.touch()
        self.assertEqual(self._run('--add-badexp'), 0)
        exposures, _ = self._tables()
        self.assertEqual(exposures['EXPID'].tolist(), [100])

    def test_bad_only_fresh_output_and_update(self):
        write_exptable_fixture(self.prod, 20211001, 100, laststep='skysub')
        real_findfile = mod.findfile
        with patch.object(mod, 'findfile', side_effect=lambda kind, **kw:
                          ('/missing', False) if kind.startswith('fiberassign') else real_findfile(kind, **kw)):
            self.assertEqual(self._run('--add-badexp', '--cameras', 'b0'), 0)
            self._add_exposure(expid=200, laststep='skysub')
            self.assertEqual(self._run('--add-badexp', '--expids', '200', '--cameras', 'b0'), 0)
        exposures, frames = self._tables()
        self.assertEqual(exposures['EXPID'].tolist(), [100, 200])
        self.assertTrue(np.all(exposures['TSNR2_ELG'] == 0))
        self.assertEqual(frames['CAMERA'].tolist(), ['b0', 'b0'])

    def test_bad_only_rank_is_included(self):
        write_exptable_fixture(self.prod, 20211001, 100, laststep='skysub')
        self._mpi([mod._empty_tables()])
        real_findfile = mod.findfile
        with patch.object(mod, 'findfile', side_effect=lambda kind, **kw:
                          ('/missing', False) if kind.startswith('fiberassign') else real_findfile(kind, **kw)):
            self.assertEqual(self._run('--mpi', '--add-badexp'), 0)
        self.assertEqual(self._tables()[0]['EXPID'].tolist(), [100])

    def test_rerun_preserves_gfa_inputs_and_recomputes_efftimes(self):
        # GFA inputs survive an update without --gfa-proc-dir, and effective
        # times are recomputed from them with the current sky magnitudes.
        self._run()
        exposures, frames = self._tables()
        exposures['TRANSPARENCY_GFA'] = [.9]
        exposures['EFFTIME_GFA'] = [750.]
        mod.write_output(exposures, frames, str(self.outfile))
        def efftime(table):
            value = np.asarray(table['SKY_MAG_R_SPEC'], dtype=float) * 10.
            return value, value, value
        with patch.object(mod, 'compute_efftime', side_effect=efftime):
            self._run('--update')
        exposures, _ = self._tables()
        self.assertAlmostEqual(float(exposures['TRANSPARENCY_GFA'][0]), .9, places=6)
        self.assertAlmostEqual(float(exposures['EFFTIME_GFA'][0]), 210., places=4)

    def test_unreadable_existing_summary_is_not_replaced(self):
        # FRAMES present but EXPOSURES missing: refuse to merge, keep the file
        self._run()
        _, frames = self._tables()
        fits.HDUList([fits.PrimaryHDU(), fits.table_to_hdu(frames)]).writeto(str(self.outfile), overwrite=True)
        before = self.outfile.read_bytes()
        self._add_exposure(night=20211002, expid=200)
        self.assertEqual(self._run('--nights', '20211002'), 1)
        self.assertEqual(self.outfile.read_bytes(), before)
        self.assertEqual(self._run('--nights', '20211002', '--overwrite'), 0)

    def test_frames_without_exposures_is_not_merged(self):
        self._run()
        exposures, frames = self._tables()
        mod.write_output(exposures[:0], frames, str(self.outfile))
        before = self.outfile.read_bytes()
        self.assertEqual(self._run(), 1)
        self.assertEqual(self.outfile.read_bytes(), before)

    def _no_fiberassign(self):
        real_findfile = mod.findfile
        return patch.object(mod, 'findfile', side_effect=lambda kind, **kw:
                            ('/missing', False) if kind.startswith('fiberassign') else real_findfile(kind, **kw))

    def test_bad_exposure_removed_without_add_badexp(self):
        # Changing LASTSTEP to ignore later is the same as ignoring it from the start
        self._add_exposure(expid=101)
        self._run()
        write_exptable_fixture(self.prod, 20211001, 100, laststep='ignore')
        self.assertEqual(self._run(), 0)
        exposures, frames = self._tables()
        self.assertEqual(exposures['EXPID'].tolist(), [101])
        self.assertEqual(set(frames['EXPID'].tolist()), {101})

    def test_all_cameras_bad_removed_without_add_badexp(self):
        self._add_exposure(expid=101)
        self._run()
        write_exptable_fixture(self.prod, 20211001, 100, BADCAMWORD='a0')
        self.assertEqual(self._run(), 0)
        exposures, frames = self._tables()
        self.assertEqual(exposures['EXPID'].tolist(), [101])
        self.assertEqual(set(frames['EXPID'].tolist()), {101})

    def test_removed_bad_exposure_updates_tiles(self):
        # tile 1234 loses its only exposure and is dropped; tile 5678 is recomputed
        tiles = self.prod / 'tiles.fits'
        self._add_exposure(expid=101, tileid=5678)
        self._add_exposure(expid=102, tileid=5678)
        with self._tile_capture():
            self._run('--tile-completeness', str(tiles))
        self.assertEqual(sorted(Table.read(str(tiles))['TILEID'].tolist()), [1234, 5678])
        write_exptable_fixture(self.prod, 20211001, 100, laststep='ignore')
        write_exptable_fixture(self.prod, 20211001, 102, tileid=5678, laststep='ignore')
        def merge(previous, new):
            return vstack([previous[~np.isin(previous['TILEID'], new['TILEID'])], new])
        with self._tile_capture() as capture, patch.object(mod, 'merge_tile_completeness_table', side_effect=merge):
            self.assertEqual(self._run('--tile-completeness', str(tiles)), 0)
        self.assertEqual(capture.call_args[0][0]['EXPID'].tolist(), [101])
        self.assertEqual(Table.read(str(tiles))['TILEID'].tolist(), [5678])

    def test_removing_last_exposure_writes_empty_output(self):
        # removal of every row is intentional: write the empty summary and drop the tile
        tiles = self.prod / 'tiles.fits'
        with self._tile_capture():
            self._run('--tile-completeness', str(tiles))
        write_exptable_fixture(self.prod, 20211001, 100, laststep='ignore')
        def merge(previous, new):
            return vstack([previous[~np.isin(previous['TILEID'], new['TILEID'])], new])
        with self._tile_capture(), patch.object(mod, 'merge_tile_completeness_table', side_effect=merge):
            self.assertEqual(self._run('--tile-completeness', str(tiles)), 0)
        exposures, frames = self._tables()
        self.assertEqual(len(exposures), 0)
        self.assertEqual(len(frames), 0)
        self.assertEqual(len(Table.read(str(tiles))), 0)

    def test_empty_tile_csv_can_be_repopulated(self):
        # remove the last exposure with a CSV tile file, then add a new exposure
        tiles = self.prod / 'tiles.csv'
        def compute(exposures, *args, **kwargs):
            ids = np.unique(exposures['TILEID'])
            return Table({'TILEID': ids, 'SURVEY': np.full(len(ids), 'main'),
                          'GOALTIME': np.full(len(ids), 1000.)})
        def merge(previous, new):
            return vstack([previous[~np.isin(previous['TILEID'], new['TILEID'])], new])
        with patch.object(mod, 'compute_tile_completeness_table', side_effect=compute), \
             patch.object(mod, 'merge_tile_completeness_table', side_effect=merge):
            self._run('--tile-completeness', str(tiles))
            write_exptable_fixture(self.prod, 20211001, 100, laststep='ignore')
            self.assertEqual(self._run('--tile-completeness', str(tiles)), 0)
            self.assertEqual(len(Table.read(str(tiles))), 0)
            self._add_exposure(night=20211002, expid=200, tileid=5678)
            self.assertEqual(self._run('--tile-completeness', str(tiles)), 0)
        self.assertEqual(Table.read(str(tiles))['TILEID'].tolist(), [5678])

    def test_all_cameras_bad_replaces_existing_rows(self):
        # CAMWORD == BADCAMWORD makes the exposure a bad exposure
        self._run()
        write_exptable_fixture(self.prod, 20211001, 100, BADCAMWORD='a0')
        real_findfile = mod.findfile
        with patch.object(mod, 'findfile', side_effect=lambda kind, **kw:
                          ('/missing', False) if kind.startswith('fiberassign') else real_findfile(kind, **kw)):
            self.assertEqual(self._run('--add-badexp'), 0)
        exposures, frames = self._tables()
        self.assertEqual(float(exposures['TSNR2_ELG'][0]), 0.)
        self.assertTrue(np.all(frames['TSNR2_ELG'] == 0.))

    def test_invalidated_gfa_clears_historical_efftimes(self):
        self._run()
        exposures, frames = self._tables()
        exposures['TRANSPARENCY_GFA'] = [.9]
        for col in ('EFFTIME_GFA', 'EFFTIME_DARK_GFA', 'EFFTIME_BRIGHT_GFA', 'EFFTIME_BACKUP_GFA'):
            exposures[col] = [750.]
        mod.write_output(exposures, frames, str(self.outfile))
        self._add_exposure(night=20211002, expid=200)
        gfa = Table({'EXPID': [100], 'TRANSPARENCY': [np.nan]})
        with patch.object(mod, 'read_gfa_data', return_value=gfa):
            self._run('--nights', '20211002', '--gfa-proc-dir', '/fake')
        exposures, _ = self._tables()
        for col in ('EFFTIME_GFA', 'EFFTIME_DARK_GFA', 'EFFTIME_BRIGHT_GFA', 'EFFTIME_BACKUP_GFA'):
            self.assertEqual(float(exposures[col][exposures['EXPID'] == 100][0]), 0.)

    def test_expids_tile_selection_includes_all_exposures_of_tile(self):
        self._add_exposure(expid=200)
        self._add_exposure(expid=300, tileid=4321)
        self._run()
        with self._tile_capture() as compute:
            self._run('--expids', '100', '--tile-completeness', str(self.prod / 'tiles.fits'))
        self.assertEqual(compute.call_args.args[0]['EXPID'].tolist(), [100, 200])

    def test_delayed_gfa_adds_old_tiles_to_requested_night(self):
        self._run()
        self._add_exposure(night=20211002, expid=200, tileid=4321)
        gfa = Table({'EXPID': [100], 'TRANSPARENCY': [.9]})
        with self._tile_capture() as compute, patch.object(mod, 'read_gfa_data', return_value=gfa):
            self._run('--nights', '20211002', '--gfa-proc-dir', '/fake',
                      '--tile-completeness', str(self.prod / 'tiles.fits'))
        self.assertEqual(compute.call_args.args[0]['EXPID'].tolist(), [100, 200])

    def test_changed_efftimes_also_refresh_historical_tiles(self):
        self._run()
        exposures, frames = self._tables()
        exposures['TRANSPARENCY_GFA'] = [.9]
        exposures['EFFTIME_GFA'] = [750.]
        mod.write_output(exposures, frames, str(self.outfile))
        self._add_exposure(night=20211002, expid=200, tileid=4321)
        with self._tile_capture() as compute, \
                patch.object(mod, 'add_gfa_columns', side_effect=lambda table, path: (table, [])), \
                patch.object(mod, 'compute_efftime', return_value=(np.array([500.]),) * 3):
            self._run('--nights', '20211002', '--gfa-proc-dir', '/fake',
                      '--tile-completeness', str(self.prod / 'tiles.fits'))
        self.assertEqual(compute.call_args.args[0]['EXPID'].tolist(), [100, 200])

    def test_legacy_skymag_file_and_compute_alias(self):
        skyfile = self.prod / 'sky.fits'
        Table({'NIGHT': [20211001], 'EXPID': [100], 'SKY_MAG_G': [23.],
               'SKY_MAG_R': [22.], 'SKY_MAG_Z': [21.]}).write(skyfile)
        self._run('--skymags', str(skyfile))
        self.assertEqual(float(self._tables()[0]['SKY_MAG_R_SPEC'][0]), 22.)
        self._run('--skymags', str(skyfile), '--recompute-skymags')
        self.skymag.assert_called()
        self.assertEqual(float(self._tables()[0]['SKY_MAG_R_SPEC'][0]), 21.)

    def test_reference_reader_accepts_modern_and_legacy_extensions(self):
        from desispec.test import test_tsnr_afterburner_regression as regression
        self._run()
        for names in [('EXPOSURES', 'FRAMES'), ('TSNR2_EXPID', 'TSNR2_FRAME')]:
            exposures, frames = self._tables()
            ref = self.prod / 'reference.fits'
            fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(exposures, name=names[0]),
                          fits.BinTableHDU(frames, name=names[1])]).writeto(ref, overwrite=True)
            with patch.object(regression, '_V1_FITS', str(ref)):
                self.assertEqual(regression._read_reference('EXPOSURES')['EXPID'].tolist(), [100])
                self.assertEqual(len(regression._read_reference('FRAMES')), 3)

    def test_reference_comparison_regenerates_stale_output(self):
        from desispec.test import test_tsnr_afterburner_regression as regression
        self._run()
        current = self.prod / 'current.fits'
        current.write_bytes(b'stale output must not be reused')
        with patch.object(regression, '_V1_FITS', str(self.outfile)), \
                patch.object(regression, '_V2_FITS', str(current)), \
                patch.object(regression, '_V2_GENERATED', False):
            regression._ensure_v2_output()
        self.assertEqual(read_table(str(current), 'EXPOSURES')['EXPID'].tolist(), [100])


if __name__ == '__main__':
    unittest.main()
