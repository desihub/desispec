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
from astropy.table import Table

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
        comm.gather.side_effect = lambda value, root: [value, *[(frm, exp, None) for frm, exp in extra_tables]]
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
        result = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--recompute-skymags', result.stdout)

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
        comm.gather.side_effect = lambda value, root: [value, (frames, exposures, 'Rank 1: missing input')]
        self.assertEqual(self._run('--mpi'), 1)
        self.assertEqual(self.outfile.read_bytes(), before)

    def test_multiprocessing_dispatches_exposure_loader(self):
        with patch.object(mod.multiprocessing, 'Pool') as pool:
            pool.return_value.__enter__.return_value.map.side_effect = lambda fn, args: list(map(fn, args))
            self.assertEqual(self._run('--nproc', '2'), 0)
        pool.assert_called_once_with(2)
        self.assertEqual(float(self._tables()[0]['TSNR2_ELG'][0]), 30.)

    def test_missing_fallback_camera_is_skipped(self):
        # b0 has neither a QA value nor a cframe: skip it, as the original afterburner did
        self._run()
        write_qa_fixture(self.prod, 20211001, 100, missing=['TSNR2_ELG_B'])
        self.assertEqual(self._run(), 0)
        exposures, frames = self._tables()
        self.assertEqual(frames['CAMERA'].tolist(), ['r0', 'z0'])
        self.assertEqual(float(exposures['TSNR2_ELG'][0]), 20.)

    def test_unreadable_exposure_keeps_existing_rows(self):
        # LASTSTEP=all exposure with no QA and no cframes: skip it, keep the old rows
        self._run()
        before_exp, before_frames = self._tables()
        os.remove(mod.findfile('exposureqa', night=20211001, expid=100))
        self._add_exposure(expid=101)
        self.assertEqual(self._run(), 0)
        exposures, frames = self._tables()
        self.assertEqual(exposures['EXPID'].tolist(), [100, 101])
        old = exposures[exposures['EXPID'] == 100]
        self.assertEqual(float(old['TSNR2_ELG'][0]), float(before_exp['TSNR2_ELG'][0]))
        self.assertEqual(int(np.sum(frames['EXPID'] == 100)), len(before_frames))

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

    def test_rerun_preserves_gfa_without_refresh(self):
        self._run()
        exposures, frames = self._tables()
        exposures['TRANSPARENCY_GFA'] = [.9]
        exposures['EFFTIME_GFA'] = [750.]
        mod.write_output(exposures, frames, str(self.outfile))
        self._run('--update')
        exposures, _ = self._tables()
        self.assertAlmostEqual(float(exposures['TRANSPARENCY_GFA'][0]), .9, places=6)
        self.assertEqual(float(exposures['EFFTIME_GFA'][0]), 750.)

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
        self._run('--skymags', str(skyfile), '--compute-skymags')
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
