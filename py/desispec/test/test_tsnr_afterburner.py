"""
Unit tests for desispec.scripts.tsnr_afterburner.

Tests use synthetic data only and do not require NERSC or actual DESI files.
"""

import os
import tempfile
import unittest
from unittest import mock
from unittest.mock import patch, MagicMock

import numpy as np
from astropy.table import Table

from desispec.scripts.tsnr_afterburner import (
    parse,
    derive_targ_info,
    get_skymag_values,
    build_tables,
    inject_bad_exposures,
    add_gfa_columns,
    add_gfa_efftimes,
    merge_exposures,
    merge_frames,
    write_output,
    _make_metadata_row,
    _add_efftimes,
    _add_gfa_zero_cols,
    _reorder_exposures,
    _read_etc_values,
    _petalqa_camera_values,
    _fill_targ_from_header,
    find_processed_nights,
    _EXP_SUMMARY_COLUMN_ORDER,
    _TARG_DEFAULTS,
    _TSNR2_TRACERS,
    _SEPT_2021_CUTOFF,
    _CAMERA_BANDS,
    _PETALS,
)


def _mock_tsnr2_to_efftime(tsnr2, tracer):
    """Simple stand-in for tsnr2_to_efftime that does not require DESIMODEL."""
    return float(tsnr2) * 0.1


def _make_petalqa(petals, tracers=_TSNR2_TRACERS, bands=_CAMERA_BANDS, value=10.0):
    """Return a numpy structured array mimicking a PETALQA HDU.

    Args:
        petals: list of int, PETAL_LOC values.
        tracers: sequence of str tracer names.
        bands: sequence of str band letters.
        value: float, fill value for all TSNR2 columns.

    Returns:
        numpy.ndarray with structured dtype.
    """
    dtype_fields = [('PETAL_LOC', np.int16)]
    for tracer in tracers:
        for band in bands:
            dtype_fields.append(('TSNR2_{}_{}'.format(tracer, band.upper()), np.float32))
    arr = np.zeros(len(petals), dtype=dtype_fields)
    arr['PETAL_LOC'] = petals
    for tracer in tracers:
        for band in bands:
            col = 'TSNR2_{}_{}'.format(tracer, band.upper())
            arr[col] = value
    return arr


def _make_exposure_row(expid=100, night=20210601, tileid=1234, petals=None, value=10.0):
    """Build a synthetic exposure dict as returned by read_one_exposure().

    Args:
        expid: int.
        night: int.
        tileid: int.
        petals: list of int petal numbers (default [0, 1, 2]).
        value: float, fill value for TSNR2 in PETALQA.

    Returns:
        dict.
    """
    if petals is None:
        petals = [0, 1, 2]
    row = {
        'NIGHT': np.int32(night),
        'EXPID': np.int32(expid),
        'TILEID': np.int32(tileid),
        'TILERA': np.float32(180.0),
        'TILEDEC': np.float32(30.0),
        'MJD': np.float64(59366.5),
        'EXPTIME': np.float32(900.0),
        'AIRMASS': np.float32(1.1),
        'EBV': np.float32(0.05),
        'SEEING_ETC': np.float32(1.2),
        'EFFTIME_ETC': np.float32(800.0),
        'SKY_MAG_G_SPEC': np.float32(21.5),
        'SKY_MAG_R_SPEC': np.float32(20.5),
        'SKY_MAG_Z_SPEC': np.float32(19.5),
        'SURVEY': 'main',
        'FAPRGRM': 'dark',
        'FAFLAVOR': 'maindark',
        'GOALTYPE': 'dark',
        'GOALTIME': np.float32(1000.0),
        'MINTFRAC': np.float32(0.9),
        'PETALQA': _make_petalqa(petals, value=value),
    }
    # CAMERA_ROWS as read_one_exposure() builds them from PETALQA.
    row['CAMERA_ROWS'] = {}
    for petal in petals:
        for band in _CAMERA_BANDS:
            camera = band + str(petal)
            values = _petalqa_camera_values(row['PETALQA'], camera)
            if values:
                row['CAMERA_ROWS'][camera] = values
    return row


def _make_camera_row(expid=100, night=20210601, tileid=1234, camera='b0', value=10.0):
    """Build a synthetic per-camera dict as returned by read_one_camera().

    Args:
        expid: int.
        night: int.
        tileid: int.
        camera: str.
        value: float, fill value for TSNR2 keys.

    Returns:
        dict.
    """
    row = {
        'NIGHT': np.int32(night),
        'EXPID': np.int32(expid),
        'TILEID': np.int32(tileid),
        'TILERA': np.float32(180.0),
        'TILEDEC': np.float32(30.0),
        'MJD': np.float64(59366.5),
        'EXPTIME': np.float32(900.0),
        'AIRMASS': np.float32(1.1),
        'EBV': np.float32(0.05),
        'SEEING_ETC': np.float32(1.2),
        'EFFTIME_ETC': np.float32(800.0),
        'CAMERA': camera,
        'SURVEY': 'main',
        'FAPRGRM': 'dark',
        'FAFLAVOR': 'maindark',
        'GOALTYPE': 'dark',
        'GOALTIME': np.float32(1000.0),
        'MINTFRAC': np.float32(0.9),
    }
    for tracer in _TSNR2_TRACERS:
        row['TSNR2_{}'.format(tracer)] = np.float32(value)
    return row


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestConstants(unittest.TestCase):
    """Verify constants match expected values from v1."""

    def test_exp_summary_column_order_length(self):
        self.assertEqual(len(_EXP_SUMMARY_COLUMN_ORDER), 51)

    def test_sept_2021_cutoff(self):
        self.assertEqual(_SEPT_2021_CUTOFF, 20210901)

    def test_tsnr2_tracers(self):
        expected = ('ELG', 'QSO', 'LRG', 'LYA', 'BGS', 'GPBDARK', 'GPBBRIGHT', 'GPBBACKUP')
        self.assertEqual(_TSNR2_TRACERS, expected)

    def test_camera_bands(self):
        self.assertEqual(_CAMERA_BANDS, ('b', 'r', 'z'))

    def test_petals(self):
        self.assertEqual(_PETALS, list(range(10)))

    def test_targ_defaults_keys(self):
        for k in ('SURVEY', 'GOALTYPE', 'FAPRGRM', 'FAFLAVOR', 'MINTFRAC', 'GOALTIME'):
            self.assertIn(k, _TARG_DEFAULTS)

    def test_no_duplicate_columns(self):
        self.assertEqual(len(_EXP_SUMMARY_COLUMN_ORDER), len(set(_EXP_SUMMARY_COLUMN_ORDER)))


class TestParse(unittest.TestCase):
    """Test argparse setup."""

    def test_defaults(self):
        args = parse(['--prod', 'daily'])
        self.assertEqual(args.prod, 'daily')
        self.assertIsNone(args.outfile)
        self.assertIsNone(args.nights)
        self.assertIsNone(args.expids)
        self.assertFalse(args.mpi)
        self.assertEqual(args.nproc, 1)
        self.assertFalse(args.overwrite)
        self.assertFalse(args.recompute)

    def test_mpi_with_nproc_allowed(self):
        args = parse(['--mpi', '--nproc', '4'])
        self.assertTrue(args.mpi)
        self.assertEqual(args.nproc, 4)

    def test_cameras_arg(self):
        args = parse(['--cameras', 'b0,r0'])
        self.assertEqual(args.cameras, 'b0,r0')

    def test_outfile_arg(self):
        args = parse(['-o', '/tmp/test.fits'])
        self.assertEqual(args.outfile, '/tmp/test.fits')

    def test_overwrite_flag(self):
        args = parse(['--overwrite'])
        self.assertTrue(args.overwrite)

    def test_update_flag(self):
        self.assertTrue(parse([]).update)
        self.assertTrue(parse(['--update']).update)
        self.assertFalse(parse(['--no-update']).update)

    def test_output_modes_mutually_exclusive(self):
        for flags in (['--no-update', '--overwrite'], ['--update', '--overwrite']):
            with self.assertRaises(SystemExit):
                parse(flags)

    def test_recompute_skymags_flag(self):
        args = parse(['--recompute-skymags'])
        self.assertTrue(args.recompute_skymags)

    def test_old_compute_skymags_name_removed(self):
        with self.assertRaises(SystemExit):
            parse(['--compute-skymags'])


class TestDeriveTargInfo(unittest.TestCase):
    """Test derive_targ_info() survey/goaltype normalization."""

    def test_maindark_passthrough(self):
        entry = dict(_TARG_DEFAULTS)
        entry['FAFLAVOR'] = 'maindark'
        result = derive_targ_info(entry)
        self.assertEqual(result['GOALTYPE'], 'dark')

    def test_sv1_detection(self):
        entry = dict(_TARG_DEFAULTS)
        entry['FAFLAVOR'] = 'sv1elg'
        result = derive_targ_info(entry)
        self.assertEqual(result['SURVEY'], 'sv1')
        self.assertEqual(result['GOALTYPE'], 'dark')

    def test_sv2_detection(self):
        entry = dict(_TARG_DEFAULTS)
        entry['FAFLAVOR'] = 'sv2elg'
        result = derive_targ_info(entry)
        self.assertEqual(result['SURVEY'], 'sv2')

    def test_cmx_detection(self):
        entry = dict(_TARG_DEFAULTS)
        entry['FAFLAVOR'] = 'cmxlrgqso'
        result = derive_targ_info(entry)
        # cmxlrgqso maps to sv1 survey via special-case check
        self.assertEqual(result['SURVEY'], 'sv1')

    def test_mainbright_goaltype(self):
        entry = dict(_TARG_DEFAULTS)
        entry['FAFLAVOR'] = 'mainbright'
        result = derive_targ_info(entry)
        self.assertEqual(result['GOALTYPE'], 'bright')

    def test_unknown_faflavor_unchanged(self):
        entry = dict(_TARG_DEFAULTS)
        result = derive_targ_info(entry)
        self.assertEqual(result['SURVEY'], 'unknown')

    def test_missing_keys_filled(self):
        entry = {'FAFLAVOR': 'maindark'}
        result = derive_targ_info(entry)
        for key in _TARG_DEFAULTS:
            self.assertIn(key, result)

    def test_dark1b_normalized(self):
        entry = dict(_TARG_DEFAULTS)
        entry['GOALTYPE'] = 'dark1b'
        entry['FAFLAVOR'] = 'maindark'
        result = derive_targ_info(entry)
        self.assertEqual(result['GOALTYPE'], 'dark')

    def test_bright1b_normalized(self):
        entry = dict(_TARG_DEFAULTS)
        entry['GOALTYPE'] = 'bright1b'
        entry['FAFLAVOR'] = 'mainbright'
        result = derive_targ_info(entry)
        self.assertEqual(result['GOALTYPE'], 'bright')

    def test_dith_faprgrm_sets_cmx_survey(self):
        """FAPRGRM starting with 'dith' and unknown SURVEY should set SURVEY='cmx'."""
        entry = dict(_TARG_DEFAULTS)
        entry['FAFLAVOR'] = 'unknown'
        entry['FAPRGRM'] = 'dithfaint'
        result = derive_targ_info(entry)
        self.assertEqual(result['SURVEY'], 'cmx')

    def test_goaltype_inferred_from_faprgrm_qso(self):
        """With FAFLAVOR and GOALTYPE unknown, GOALTYPE is inferred from FAPRGRM keywords."""
        entry = dict(_TARG_DEFAULTS)
        entry['FAPRGRM'] = 'qso'
        result = derive_targ_info(entry)
        self.assertEqual(result['GOALTYPE'], 'dark')

    def test_goaltype_inferred_from_faprgrm_bgs(self):
        """When FAPRGRM contains 'bgs' and FAFLAVOR is unknown, GOALTYPE should be 'bright'."""
        entry = dict(_TARG_DEFAULTS)
        entry['FAPRGRM'] = 'bgsany'
        result = derive_targ_info(entry)
        self.assertEqual(result['GOALTYPE'], 'bright')

    def test_known_faflavor_overrides_goaltype(self):
        """A known FAFLAVOR sets GOALTYPE via faflavor2program, as in the original afterburner."""
        entry = dict(_TARG_DEFAULTS)
        entry['FAFLAVOR'] = 'specialtertiary42'
        entry['FAPRGRM'] = 'tertiary42'
        entry['GOALTYPE'] = 'dark'
        result = derive_targ_info(entry)
        self.assertEqual(result['GOALTYPE'], 'other')

    def test_known_faflavor_other_not_inferred_from_faprgrm(self):
        """faflavor2program 'other' is kept even if FAPRGRM looks dark."""
        entry = dict(_TARG_DEFAULTS)
        entry['FAFLAVOR'] = 'other'
        entry['FAPRGRM'] = 'qso'
        result = derive_targ_info(entry)
        self.assertEqual(result['GOALTYPE'], 'other')

    def test_unknown_faflavor_keeps_goaltype(self):
        """Without FAFLAVOR, an already known GOALTYPE is kept."""
        entry = dict(_TARG_DEFAULTS)
        entry['GOALTYPE'] = 'bright'
        result = derive_targ_info(entry)
        self.assertEqual(result['GOALTYPE'], 'bright')


class TestGetSkymag(unittest.TestCase):
    """Test get_skymag_values() wrapper."""

    @patch('desispec.scripts.tsnr_afterburner.compute_skymag', return_value=(22.0, 21.5, 19.0))
    def test_returns_dict(self, mock_skymag):
        result = get_skymag_values(20210601, 12345)
        self.assertIn('SKY_MAG_G_SPEC', result)
        self.assertIn('SKY_MAG_R_SPEC', result)
        self.assertIn('SKY_MAG_Z_SPEC', result)
        self.assertAlmostEqual(result['SKY_MAG_G_SPEC'], 22.0, places=5)

    @patch('desispec.scripts.tsnr_afterburner.compute_skymag', side_effect=RuntimeError('no data'))
    def test_exception_returns_99(self, mock_skymag):
        result = get_skymag_values(20210601, 99999)
        self.assertEqual(result['SKY_MAG_G_SPEC'], 99.0)
        self.assertEqual(result['SKY_MAG_R_SPEC'], 99.0)
        self.assertEqual(result['SKY_MAG_Z_SPEC'], 99.0)


class TestReadOneExposureSkymag(unittest.TestCase):
    """Test sky mag branching logic in read_one_exposure()."""

    def _make_fitsio_mock(self, has_skymags=True, tileid=1234):
        """Build a minimal fitsio mock returning synthetic headers/data."""
        hdr = {
            'TILEID': tileid, 'TILERA': 180.0, 'TILEDEC': 30.0,
            'MJD-OBS': 59366.5, 'EXPTIME': 900.0, 'AIRMASS': 1.1,
            'ACQFWHM': 1.2, 'ETCTEFF': 800.0,
            'SURVEY': 'main', 'FAPRGRM': 'dark', 'FAFLAVOR': 'maindark',
            'GOALTYPE': 'dark', 'GOALTIME': 1000.0, 'MINTFRAC': 0.9,
        }
        if has_skymags:
            hdr['SKY_MAG_G_SPEC'] = 22.0
            hdr['SKY_MAG_R_SPEC'] = 21.0
            hdr['SKY_MAG_Z_SPEC'] = 19.0
        return hdr

    def _fake_fitsio_read(self, filename, hdu, columns=None):
        """Return HDU-appropriate synthetic data for fitsio.read calls."""
        if hdu == 'FIBERQA':
            ebv = np.zeros(10, dtype=[('EBV', np.float32)])
            ebv['EBV'] = 0.05
            return ebv
        if hdu == 'PETALQA':
            return _make_petalqa([0, 1, 2])
        raise ValueError('Unexpected HDU in fitsio.read mock: {}'.format(hdu))

    @patch('desispec.scripts.tsnr_afterburner.findfile')
    def test_uses_header_skymags_when_present(self, mock_ff):
        mock_ff.return_value = '/fake/path'
        with patch('os.path.isfile', return_value=True), \
             patch('desispec.scripts.tsnr_afterburner.fitsio.read_header',
                   return_value=self._make_fitsio_mock(has_skymags=True)), \
             patch('desispec.scripts.tsnr_afterburner.fitsio.read',
                   side_effect=self._fake_fitsio_read):
            from desispec.scripts.tsnr_afterburner import read_one_exposure
            result = read_one_exposure(20210601, 12345, recompute_skymags=False)
        self.assertIsNotNone(result, 'read_one_exposure returned None — check mock setup')
        self.assertIn('SKY_MAG_G_SPEC', result)
        self.assertAlmostEqual(float(result['SKY_MAG_G_SPEC']), 22.0, places=4)
        self.assertAlmostEqual(float(result['SKY_MAG_R_SPEC']), 21.0, places=4)
        self.assertAlmostEqual(float(result['SKY_MAG_Z_SPEC']), 19.0, places=4)

    @patch('desispec.scripts.tsnr_afterburner.get_skymag_values',
           return_value={'SKY_MAG_G_SPEC': 99.0, 'SKY_MAG_R_SPEC': 99.0, 'SKY_MAG_Z_SPEC': 99.0})
    @patch('desispec.scripts.tsnr_afterburner.findfile')
    def test_falls_back_to_computed_skymags_when_missing(self, mock_ff, mock_skymag):
        """When sky-mag header keywords are absent, get_skymag_values() should be called."""
        mock_ff.return_value = '/fake/path'
        with patch('os.path.isfile', return_value=True), \
             patch('desispec.scripts.tsnr_afterburner.fitsio.read_header',
                   return_value=self._make_fitsio_mock(has_skymags=False)), \
             patch('desispec.scripts.tsnr_afterburner.fitsio.read',
                   side_effect=self._fake_fitsio_read):
            from desispec.scripts.tsnr_afterburner import read_one_exposure
            result = read_one_exposure(20210601, 12345, recompute_skymags=False)
        self.assertIsNotNone(result, 'read_one_exposure returned None — check mock setup')
        mock_skymag.assert_called_once()
        self.assertAlmostEqual(float(result['SKY_MAG_G_SPEC']), 99.0, places=4)


class TestTsnr2Aggregation(unittest.TestCase):
    """Test that per-band TSNR2 from PETALQA is correctly summed and averaged."""

    def setUp(self):
        patcher = mock.patch('desispec.scripts.tsnr_afterburner.tsnr2_to_efftime',
                             side_effect=_mock_tsnr2_to_efftime)
        self.mock_efftime = patcher.start()
        self.addCleanup(patcher.stop)

    def test_single_petal_three_bands_sums_correctly(self):
        """TSNR2 per-petal = sum over bands, per-exposure = mean over petals."""
        # one exposure, one petal, 3 bands, value=10 in each band
        exp_row = _make_exposure_row(expid=1, petals=[0], value=10.0)
        # cameras for petal 0: b0, r0, z0
        camword_map = {1: ['b0', 'r0', 'z0']}
        frames, exposures = build_tables([exp_row], camword_map=camword_map)

        # Each FRAMES row has TSNR2_ELG = 10 (per-band value from PETALQA)
        self.assertEqual(len(frames), 3)
        for row in frames:
            self.assertAlmostEqual(float(row['TSNR2_ELG']), 10.0, places=3)

        # EXPOSURES: petal sum = 30, mean over 1 petal = 30
        self.assertEqual(len(exposures), 1)
        self.assertAlmostEqual(float(exposures['TSNR2_ELG'][0]), 30.0, places=3)

    def test_two_petals_mean(self):
        """Mean over petals with equal values should equal per-petal sum."""
        exp_row = _make_exposure_row(expid=2, petals=[0, 1], value=10.0)
        camword_map = {2: ['b0', 'r0', 'z0', 'b1', 'r1', 'z1']}
        frames, exposures = build_tables([exp_row], camword_map=camword_map)

        # Both petals sum to 30 each, mean = 30
        self.assertAlmostEqual(float(exposures['TSNR2_ELG'][0]), 30.0, places=3)

    def test_missing_petal_in_petalqa_ignored(self):
        """Camera referencing a petal not in PETALQA should be silently ignored."""
        exp_row = _make_exposure_row(expid=3, petals=[0], value=10.0)
        # request petal 0 and petal 1 cameras but petalqa only has petal 0
        camword_map = {3: ['b0', 'r0', 'z0', 'b1', 'r1', 'z1']}
        frames, exposures = build_tables([exp_row], camword_map=camword_map)

        # petal 1 rows should not be in frames
        for row in frames:
            self.assertIn(row['CAMERA'], ['b0', 'r0', 'z0'])

    def test_cameras_filter_applied(self):
        """With cameras filtered, only matching cameras produce frames rows."""
        exp_row = _make_exposure_row(expid=4, petals=[0, 1], value=10.0)
        # only b-band cameras
        camword_map = {4: ['b0', 'b1']}
        frames, exposures = build_tables([exp_row], camword_map=camword_map)

        self.assertEqual(len(frames), 2)
        for row in frames:
            self.assertTrue(row['CAMERA'].startswith('b'))


class TestBuildTables(unittest.TestCase):
    """Integration tests for build_tables()."""

    def setUp(self):
        patcher = mock.patch('desispec.scripts.tsnr_afterburner.tsnr2_to_efftime',
                             side_effect=_mock_tsnr2_to_efftime)
        self.mock_efftime = patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_path_column_order(self):
        """EXPOSURES table must have columns in _EXP_SUMMARY_COLUMN_ORDER."""
        exp_row = _make_exposure_row(expid=10)
        camword_map = {10: ['b0', 'r0', 'z0']}
        frames, exposures = build_tables([exp_row], camword_map=camword_map)

        for col in _EXP_SUMMARY_COLUMN_ORDER:
            self.assertIn(col, exposures.colnames,
                          'Missing column {} in EXPOSURES'.format(col))
        # order check
        idx = {col: i for i, col in enumerate(exposures.colnames)}
        for i, col in enumerate(_EXP_SUMMARY_COLUMN_ORDER):
            if col in idx:
                self.assertEqual(idx[col], i,
                                 'Column {} at wrong position'.format(col))

    def test_empty_input_returns_empty_tables(self):
        """Empty input should return empty tables without errors."""
        frames, exposures = build_tables([], camword_map={})
        self.assertEqual(len(frames), 0)
        self.assertEqual(len(exposures), 0)

    def test_none_row_skipped(self):
        """None entries in exposure_rows should be silently skipped."""
        exp_row = _make_exposure_row(expid=30)
        camword_map = {30: ['b0']}
        frames, exposures = build_tables([None, exp_row, None], camword_map=camword_map)
        self.assertEqual(len(exposures), 1)

    def test_efftime_spec_dark_before_sept2021(self):
        """Before SEPT 2021 cutoff, EFFTIME_SPEC should use ELG_EFFTIME_DARK."""
        exp_row = _make_exposure_row(expid=40, night=20210801)  # < cutoff
        camword_map = {40: ['b0', 'r0', 'z0']}
        _, exposures = build_tables([exp_row], camword_map=camword_map)
        self.assertAlmostEqual(
            float(exposures['EFFTIME_SPEC'][0]),
            float(exposures['ELG_EFFTIME_DARK'][0]),
            places=3)

    def test_efftime_spec_dark_after_sept2021(self):
        """After SEPT 2021 cutoff, EFFTIME_SPEC should use LRG_EFFTIME_DARK."""
        exp_row = _make_exposure_row(expid=41, night=20211001)  # > cutoff
        camword_map = {41: ['b0', 'r0', 'z0']}
        _, exposures = build_tables([exp_row], camword_map=camword_map)
        self.assertAlmostEqual(
            float(exposures['EFFTIME_SPEC'][0]),
            float(exposures['LRG_EFFTIME_DARK'][0]),
            places=3)

    def test_efftime_spec_bright_uses_bgs(self):
        """For bright GOALTYPE, EFFTIME_SPEC should use BGS_EFFTIME_BRIGHT."""
        exp_row = _make_exposure_row(expid=42, night=20211001)
        exp_row['GOALTYPE'] = 'bright'
        exp_row['FAFLAVOR'] = 'mainbright'
        camword_map = {42: ['b0', 'r0', 'z0']}
        _, exposures = build_tables([exp_row], camword_map=camword_map)
        self.assertAlmostEqual(
            float(exposures['EFFTIME_SPEC'][0]),
            float(exposures['BGS_EFFTIME_BRIGHT'][0]),
            places=3)

    def test_extname_set(self):
        """Tables must have EXTNAME metadata set."""
        exp_row = _make_exposure_row(expid=50)
        camword_map = {50: ['b0']}
        frames, exposures = build_tables([exp_row], camword_map=camword_map)
        self.assertEqual(frames.meta.get('EXTNAME'), 'FRAMES')
        self.assertEqual(exposures.meta.get('EXTNAME'), 'EXPOSURES')

    def test_extra_column_appended_after_standard_columns(self):
        """Columns not in _EXP_SUMMARY_COLUMN_ORDER should appear after all standard columns."""
        import desispec.scripts.tsnr_afterburner as _mod
        exp_row = _make_exposure_row(expid=60)
        # Temporarily shrink the column order so TSNR2_ELG (and others) become 'extra'
        original_order = _mod._EXP_SUMMARY_COLUMN_ORDER
        short_order = ['NIGHT', 'EXPID', 'TILEID', 'EXPTIME']
        _mod._EXP_SUMMARY_COLUMN_ORDER = short_order
        try:
            camword_map = {60: ['b0', 'r0', 'z0']}
            _, exposures = build_tables([exp_row], camword_map=camword_map)
        finally:
            _mod._EXP_SUMMARY_COLUMN_ORDER = original_order

        # Standard columns must come first in the declared order
        for i, col in enumerate(short_order):
            self.assertEqual(exposures.colnames[i], col,
                             'Column {} not at expected position {}'.format(col, i))
        # Extra columns are appended after the standard block
        extra_start = len(short_order)
        for col in exposures.colnames[extra_start:]:
            self.assertNotIn(col, short_order)


class TestInjectBadExposures(unittest.TestCase):
    """Test inject_bad_exposures() zero-fills bad exposures correctly."""

    def setUp(self):
        patcher = mock.patch('desispec.scripts.tsnr_afterburner.tsnr2_to_efftime',
                             side_effect=_mock_tsnr2_to_efftime)
        self.mock_efftime = patcher.start()
        self.addCleanup(patcher.stop)

        # findfile('fiberassignsvn', ...) requires FIBER_ASSIGN_DIR; patch it
        # to return a non-existent path so both fiberassign lookups are skipped.
        findfile_patcher = mock.patch(
            'desispec.scripts.tsnr_afterburner.findfile',
            return_value=('/fake/fiberassign', False),
        )
        self.mock_findfile = findfile_patcher.start()
        self.addCleanup(findfile_patcher.stop)

    def _make_empty_tables(self):
        """Create minimal empty exposures and frames tables with required columns."""
        exp_row = _make_exposure_row(expid=100)
        camword_map = {100: ['b0', 'r0', 'z0']}
        frames, exposures = build_tables([exp_row], camword_map=camword_map)
        return exposures, frames

    def test_bad_expid_added_to_exposures(self):
        """A bad EXPID not already in the table should be added."""
        exposures, frames = self._make_empty_tables()
        bad_expids = [{
            'NIGHT': 20210601, 'EXPID': 999, 'TILEID': 1234,
            'CAMWORD': 'a0123456789', 'BADCAMWORD': '',
            'EXPTIME': 600.0, 'MJD-OBS': 59366.5, 'EFFTIME_ETC': 0.0,
            'LASTSTEP': 'skysub', 'SURVEY': 'main', 'FAPRGRM': 'dark',
            'GOALTYPE': 'dark', 'GOALTIME': 1000.0, 'MINTFRAC': 0.9,
            'FAFLAVOR': 'maindark', 'EBVFAC': 1.0,
        }]
        inject_bad_exposures(exposures, frames, bad_expids)
        self.assertIn(999, exposures['EXPID'].tolist())

    def test_targeting_filled_from_fiberassign_header(self):
        """FAFLAVOR/MINTFRAC missing from the exposure table come from the fiberassign header."""
        exposures, frames = self._make_empty_tables()
        bad_expids = [{
            'NIGHT': 20260926, 'EXPID': 999, 'TILEID': 20293,
            'CAMWORD': 'a0', 'BADCAMWORD': '', 'EXPTIME': 600.0, 'MJD-OBS': 59366.5,
            'EFFTIME_ETC': 0.0, 'LASTSTEP': 'skysub', 'SURVEY': 'main', 'FAPRGRM': 'bright',
            'GOALTYPE': 'bright', 'GOALTIME': 180.0, 'MINTFRAC': 0.9, 'FAFLAVOR': 'unknown',
            'EBVFAC': 1.0,
        }]
        fa_hdr = {'TILERA': 37.362, 'TILEDEC': 1.5, 'FAFLAVOR': 'mainbright', 'FAPRGRM': 'bright',
                  'SURVEY': 'main', 'GOALTYPE': 'BRIGHT', 'MINTFRAC': 0.85, 'GOALTIME': 180.0}
        self.mock_findfile.return_value = ('/fake/fiberassign-020293.fits.gz', True)
        with patch('desispec.scripts.tsnr_afterburner.fitsio.read_header', return_value=fa_hdr):
            exposures, frames = inject_bad_exposures(exposures, frames, bad_expids)
        row = exposures[exposures['EXPID'] == 999][0]
        self.assertTrue(np.all(np.isnan(frames['TSNR2_ALPHA'][frames['EXPID'] == 999])))
        for band in ('G', 'R', 'Z'):
            self.assertTrue(np.isnan(row['SKY_MAG_{}_SPEC'.format(band)]))
        self.assertEqual(row['FAFLAVOR'], 'mainbright')
        self.assertEqual(row['PROGRAM'], 'bright')
        self.assertAlmostEqual(float(row['MINTFRAC']), 0.85, places=5)
        self.assertAlmostEqual(float(row['TILERA']), 37.362, places=3)

    def _bad_entry(self, expid=999):
        return {'NIGHT': 20260926, 'EXPID': expid, 'TILEID': 1234, 'CAMWORD': 'a0', 'BADCAMWORD': '',
                'EXPTIME': 600.0, 'MJD-OBS': 59366.5, 'EFFTIME_ETC': 0.0, 'LASTSTEP': 'skysub',
                'SURVEY': 'main', 'FAPRGRM': 'dark', 'GOALTYPE': 'dark', 'GOALTIME': 1000.0,
                'MINTFRAC': 0.9, 'FAFLAVOR': 'maindark', 'EBVFAC': 1.0}

    def test_skymags_computed_when_expdir_exists(self):
        """A bad exposure with an exposures dir gets computed sky mags, with no flag needed."""
        exposures, frames = self._make_empty_tables()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, 'exposures', '20260926', '00000999'))
            with patch('desispec.scripts.tsnr_afterburner.specprod_root', return_value=tmpdir), \
                 patch('desispec.scripts.tsnr_afterburner.get_skymag_values', return_value={
                     'SKY_MAG_G_SPEC': 22.0, 'SKY_MAG_R_SPEC': 21.0, 'SKY_MAG_Z_SPEC': 20.0}) as sky:
                exposures, _ = inject_bad_exposures(exposures, frames, [self._bad_entry()])
        sky.assert_called_once_with(20260926, 999)
        self.assertEqual(float(exposures['SKY_MAG_R_SPEC'][exposures['EXPID'] == 999][0]), 21.0)

    def test_skymags_nan_without_expdir(self):
        """A bad exposure without an exposures dir has NaN sky mags."""
        exposures, frames = self._make_empty_tables()
        with tempfile.TemporaryDirectory() as tmpdir, \
             patch('desispec.scripts.tsnr_afterburner.specprod_root', return_value=tmpdir), \
             patch('desispec.scripts.tsnr_afterburner.get_skymag_values') as sky:
            exposures, _ = inject_bad_exposures(exposures, frames, [self._bad_entry()])
        sky.assert_not_called()
        self.assertTrue(np.isnan(exposures['SKY_MAG_R_SPEC'][exposures['EXPID'] == 999][0]))

    def test_existing_expid_not_duplicated(self):
        """An EXPID already in the table should not be added again."""
        exposures, frames = self._make_empty_tables()
        initial_len = len(exposures)
        bad_expids = [{
            'NIGHT': 20210601, 'EXPID': 100,  # same as existing
            'TILEID': 1234, 'CAMWORD': 'a0', 'BADCAMWORD': '',
            'EXPTIME': 600.0, 'MJD-OBS': 59366.5, 'EFFTIME_ETC': 0.0,
            'LASTSTEP': 'skysub', 'SURVEY': 'main', 'FAPRGRM': 'dark',
            'GOALTYPE': 'dark', 'GOALTIME': 1000.0, 'MINTFRAC': 0.9,
            'FAFLAVOR': 'maindark', 'EBVFAC': 1.0,
        }]
        inject_bad_exposures(exposures, frames, bad_expids)
        self.assertEqual(len(exposures), initial_len)

    def test_tsnr2_zero_filled(self):
        """TSNR2 values for bad exposures should all be zero."""
        exposures, frames = self._make_empty_tables()
        bad_expids = [{
            'NIGHT': 20210601, 'EXPID': 888, 'TILEID': 5678,
            'CAMWORD': 'a0', 'BADCAMWORD': '',
            'EXPTIME': 600.0, 'MJD-OBS': 59366.5, 'EFFTIME_ETC': 0.0,
            'LASTSTEP': 'skysub', 'SURVEY': 'main', 'FAPRGRM': 'dark',
            'GOALTYPE': 'dark', 'GOALTIME': 1000.0, 'MINTFRAC': 0.9,
            'FAFLAVOR': 'maindark', 'EBVFAC': 1.0,
        }]
        inject_bad_exposures(exposures, frames, bad_expids)
        mask = exposures['EXPID'] == 888
        for tracer in _TSNR2_TRACERS:
            self.assertAlmostEqual(
                float(exposures['TSNR2_{}'.format(tracer)][mask][0]), 0.0, places=5)

    def test_ebvfac_converted_to_ebv(self):
        """EBVFAC > 1 should be converted to a positive EBV."""
        exposures, frames = self._make_empty_tables()
        bad_expids = [{
            'NIGHT': 20210601, 'EXPID': 777, 'TILEID': 9999,
            'CAMWORD': 'a0', 'BADCAMWORD': '',
            'EXPTIME': 600.0, 'MJD-OBS': 59366.5, 'EFFTIME_ETC': 0.0,
            'LASTSTEP': 'skysub', 'SURVEY': 'main', 'FAPRGRM': 'dark',
            'GOALTYPE': 'dark', 'GOALTIME': 1000.0, 'MINTFRAC': 0.9,
            'FAFLAVOR': 'maindark', 'EBVFAC': 1.5,
        }]
        inject_bad_exposures(exposures, frames, bad_expids)
        mask = exposures['EXPID'] == 777
        ebv = float(exposures['EBV'][mask][0])
        self.assertGreater(ebv, 0.0)

    def test_laststep_all_entry_is_zero_filled(self):
        """main() passes unreadable LASTSTEP='all' exposures here; they get zeroed rows."""
        exposures, frames = self._make_empty_tables()
        entry = self._bad_entry()
        entry['LASTSTEP'] = 'all'
        exposures, frames = inject_bad_exposures(exposures, frames, [entry])
        row = exposures[exposures['EXPID'] == 999]
        self.assertEqual(len(row), 1)
        self.assertEqual(float(row['EFFTIME_SPEC'][0]), 0.0)


class TestAddGfaColumns(unittest.TestCase):
    """Test add_gfa_columns() GFA data join logic."""

    def setUp(self):
        patcher = mock.patch('desispec.scripts.tsnr_afterburner.tsnr2_to_efftime',
                             side_effect=_mock_tsnr2_to_efftime)
        self.mock_efftime = patcher.start()
        self.addCleanup(patcher.stop)

    def _make_exposures_table(self, expids=(100, 200)):
        rows = [_make_exposure_row(expid=e) for e in expids]
        camword_map = {e: ['b0'] for e in expids}
        _, exposures = build_tables(rows, camword_map=camword_map)
        return exposures

    def _make_gfa_table(self, expids):
        n = len(expids)
        dtype = [
            ('EXPID', np.int64),
            ('NIGHT', np.int64),
            ('TRANSPARENCY', np.float64),
            ('FWHM_ASEC', np.float64),
            ('FIBER_FRACFLUX', np.float64),
            ('FIBER_FRACFLUX_ELG', np.float64),
            ('FIBER_FRACFLUX_BGS', np.float64),
            ('FIBERFAC', np.float64),
            ('FIBERFAC_ELG', np.float64),
            ('FIBERFAC_BGS', np.float64),
            ('SKY_MAG_AB', np.float64),
            ('AIRMASS', np.float64),
        ]
        arr = np.zeros(n, dtype=dtype)
        arr['EXPID'] = expids
        arr['TRANSPARENCY'] = 0.9
        arr['FWHM_ASEC'] = 1.1
        arr['FIBER_FRACFLUX'] = 0.6
        arr['FIBERFAC'] = 1.0
        arr['AIRMASS'] = 1.1
        return Table(arr)

    @patch('desispec.scripts.tsnr_afterburner.read_gfa_data')
    def test_gfa_columns_added(self, mock_gfa):
        """GFA columns should be added to exposures table."""
        exposures = self._make_exposures_table(expids=[100, 200])
        gfa = self._make_gfa_table([100, 200])
        mock_gfa.return_value = gfa

        add_gfa_columns(exposures, '/fake/gfa')
        self.assertIn('SEEING_GFA', exposures.colnames)
        self.assertIn('TRANSPARENCY_GFA', exposures.colnames)

    @patch('desispec.scripts.tsnr_afterburner.read_gfa_data')
    def test_nan_replaced_with_zero(self, mock_gfa):
        """NaN values in GFA table should be replaced with 0."""
        exposures = self._make_exposures_table(expids=[100])
        gfa = self._make_gfa_table([100])
        gfa['TRANSPARENCY'][0] = np.nan
        mock_gfa.return_value = gfa

        add_gfa_columns(exposures, '/fake/gfa')
        self.assertEqual(float(exposures['TRANSPARENCY_GFA'][0]), 0.0)

    @patch('desispec.scripts.tsnr_afterburner.read_gfa_data')
    def test_no_match_returns_empty_list(self, mock_gfa):
        """When no EXPIDs match, returns an empty list."""
        exposures = self._make_exposures_table(expids=[100])
        gfa = self._make_gfa_table([999])  # no match
        mock_gfa.return_value = gfa

        _, changed = add_gfa_columns(exposures, '/fake/gfa')
        self.assertEqual(changed, [])

    @patch('desispec.scripts.tsnr_afterburner.read_gfa_data')
    def test_fwhm_renamed_to_seeing_gfa(self, mock_gfa):
        """FWHM_ASEC must be stored as SEEING_GFA, not FWHM_ASEC_GFA."""
        exposures = self._make_exposures_table(expids=[100])
        gfa = self._make_gfa_table([100])
        mock_gfa.return_value = gfa

        add_gfa_columns(exposures, '/fake/gfa')
        self.assertIn('SEEING_GFA', exposures.colnames)
        self.assertNotIn('FWHM_ASEC_GFA', exposures.colnames)


class TestAddGfaEfftimes(unittest.TestCase):
    """Test add_gfa_efftimes()."""

    def setUp(self):
        patcher = mock.patch('desispec.scripts.tsnr_afterburner.tsnr2_to_efftime',
                             side_effect=_mock_tsnr2_to_efftime)
        self.mock_efftime = patcher.start()
        self.addCleanup(patcher.stop)

    def _make_exposures_with_gfa(self, goaltype='dark'):
        rows = [_make_exposure_row(expid=100)]
        rows[0]['GOALTYPE'] = goaltype
        if goaltype == 'bright':
            rows[0]['FAFLAVOR'] = 'mainbright'
        camword_map = {100: ['b0', 'r0', 'z0']}
        _, exposures = build_tables(rows, camword_map=camword_map)

        # inject synthetic GFA columns
        exposures['TRANSPARENCY_GFA'] = np.array([0.9])
        exposures['FIBERFAC_GFA'] = np.array([1.0])
        exposures['FIBERFAC_ELG_GFA'] = np.array([1.0])
        exposures['FIBERFAC_BGS_GFA'] = np.array([1.0])
        exposures['FIBER_FRACFLUX_GFA'] = np.array([0.6])
        exposures['FIBER_FRACFLUX_ELG_GFA'] = np.array([0.6])
        exposures['FIBER_FRACFLUX_BGS_GFA'] = np.array([0.6])
        exposures['SKY_MAG_R_SPEC'] = np.array([21.0])
        exposures['AIRMASS_GFA'] = np.array([1.1])
        exposures['SEEING_GFA'] = np.array([1.1])
        return exposures

    @patch('desispec.scripts.tsnr_afterburner.compute_efftime',
           return_value=(np.array([500.123456789]), np.array([400.0]), np.array([300.0])))
    def test_efftimes_are_float64(self, mock_ce):
        """GFA efftimes stay float64, as stored in production files, to avoid rounding changes."""
        exposures = self._make_exposures_with_gfa('dark')
        exposures = add_gfa_efftimes(exposures)
        for col in ('EFFTIME_DARK_GFA', 'EFFTIME_BRIGHT_GFA', 'EFFTIME_BACKUP_GFA', 'EFFTIME_GFA'):
            self.assertEqual(exposures[col].dtype, np.float64)
        self.assertEqual(float(exposures['EFFTIME_DARK_GFA'][0]), 500.123456789)

    def test_unknown_sky_gives_zero_efftime(self):
        """SKY_MAG_R_SPEC=99 (unknown) or NaN must not produce a GFA effective time."""
        for skymag in (99.0, np.nan):
            exposures = self._make_exposures_with_gfa('dark')
            exposures['SKY_MAG_R_SPEC'] = [skymag]
            with patch('desispec.scripts.tsnr_afterburner.compute_efftime') as ce:
                exposures = add_gfa_efftimes(exposures)
            ce.assert_not_called()
            self.assertEqual(float(exposures['EFFTIME_GFA'][0]), 0.0)

    def test_gfa_zero_cols_are_float64(self):
        row = _add_gfa_zero_cols({})
        self.assertEqual(np.asarray(row['TRANSPARENCY_GFA']).dtype, np.float64)

    @patch('desispec.scripts.tsnr_afterburner.compute_efftime',
           return_value=(np.array([500.0]), np.array([400.0]), np.array([300.0])))
    def test_efftimes_added(self, mock_ce):
        """EFFTIME_DARK/BRIGHT/BACKUP_GFA columns should be added."""
        exposures = self._make_exposures_with_gfa('dark')
        add_gfa_efftimes(exposures)
        self.assertIn('EFFTIME_DARK_GFA', exposures.colnames)
        self.assertIn('EFFTIME_BRIGHT_GFA', exposures.colnames)
        self.assertIn('EFFTIME_BACKUP_GFA', exposures.colnames)
        self.assertIn('EFFTIME_GFA', exposures.colnames)

    @patch('desispec.scripts.tsnr_afterburner.compute_efftime',
           return_value=(np.array([500.0]), np.array([400.0]), np.array([300.0])))
    def test_dark_efftime_gfa_uses_dark(self, mock_ce):
        """For dark program, EFFTIME_GFA should equal EFFTIME_DARK_GFA."""
        exposures = self._make_exposures_with_gfa('dark')
        add_gfa_efftimes(exposures)
        self.assertAlmostEqual(
            float(exposures['EFFTIME_GFA'][0]),
            float(exposures['EFFTIME_DARK_GFA'][0]),
            places=3)

    @patch('desispec.scripts.tsnr_afterburner.compute_efftime',
           return_value=(np.array([500.0]), np.array([400.0]), np.array([300.0])))
    def test_bright_efftime_gfa_uses_bright(self, mock_ce):
        """For bright program, EFFTIME_GFA should equal EFFTIME_BRIGHT_GFA."""
        exposures = self._make_exposures_with_gfa('bright')
        add_gfa_efftimes(exposures)
        self.assertAlmostEqual(
            float(exposures['EFFTIME_GFA'][0]),
            float(exposures['EFFTIME_BRIGHT_GFA'][0]),
            places=3)

    @patch('desispec.scripts.tsnr_afterburner.compute_efftime',
           return_value=(np.array([500.0]), np.array([400.0]), np.array([300.0])))
    def test_backup_efftime_gfa_uses_backup(self, mock_ce):
        """For backup program, EFFTIME_GFA should equal EFFTIME_BACKUP_GFA."""
        exposures = self._make_exposures_with_gfa('dark')
        exposures['GOALTYPE'] = ['backup']
        add_gfa_efftimes(exposures)
        self.assertAlmostEqual(
            float(exposures['EFFTIME_GFA'][0]),
            float(exposures['EFFTIME_BACKUP_GFA'][0]),
            places=3)

    def test_missing_columns_returns_table_unchanged(self):
        """When required GFA columns are absent, table is returned without EFFTIME_*_GFA columns."""
        # Use a minimal table that has none of the required GFA columns so the
        # early-return guard triggers.  (A table from build_tables would already
        # have these columns because _add_gfa_zero_cols pre-populates them.)
        minimal = Table({'EXPID': [100], 'EXPTIME': [900.0]})
        result = add_gfa_efftimes(minimal)
        self.assertNotIn('EFFTIME_DARK_GFA', result.colnames)


class TestMergeExposures(unittest.TestCase):
    """Test merge_exposures() upsert behavior."""

    def setUp(self):
        patcher = mock.patch('desispec.scripts.tsnr_afterburner.tsnr2_to_efftime',
                             side_effect=_mock_tsnr2_to_efftime)
        self.mock_efftime = patcher.start()
        self.addCleanup(patcher.stop)

    def _make_table(self, expids, value=1.0):
        rows = [_make_exposure_row(expid=e) for e in expids]
        camword_map = {e: ['b0'] for e in expids}
        _, table = build_tables(rows, camword_map=camword_map)
        return table

    def test_new_replaces_existing(self):
        """Rows in new_table should replace matching rows in existing."""
        existing = self._make_table([100, 200])
        new = self._make_table([100])  # replace row 100
        merged = merge_exposures(existing, new)

        self.assertEqual(sorted(merged['EXPID'].tolist()), [100, 200])

    def test_new_adds_when_not_in_existing(self):
        """Rows in new_table not in existing should be added."""
        existing = self._make_table([100])
        new = self._make_table([200])
        merged = merge_exposures(existing, new)

        self.assertIn(100, merged['EXPID'].tolist())
        self.assertIn(200, merged['EXPID'].tolist())

    def test_result_sorted_by_expid(self):
        """Merged table should be sorted by EXPID."""
        existing = self._make_table([200, 300])
        new = self._make_table([100])
        merged = merge_exposures(existing, new)

        expids = merged['EXPID'].tolist()
        self.assertEqual(expids, sorted(expids))

    def test_new_column_in_new_table_added_to_result(self):
        """A column present only in new_table should appear in the merged result."""
        existing = self._make_table([100])
        new = self._make_table([200])
        new['EXTRA_COL'] = np.zeros(len(new), dtype=np.float32)
        merged = merge_exposures(existing, new)
        self.assertIn('EXTRA_COL', merged.colnames)
        # row from existing should have been zero-filled for the new column
        mask = merged['EXPID'] == 100
        self.assertAlmostEqual(float(merged['EXTRA_COL'][mask][0]), 0.0)


class TestMergeFrames(unittest.TestCase):
    """Test merge_frames() upsert behavior."""

    def _make_table(self, expids, cameras=('b0',)):
        rows = []
        for e in expids:
            for c in cameras:
                rows.append(_make_camera_row(expid=e, camera=c))
        table = Table(rows=rows)
        table.meta['EXTNAME'] = 'FRAMES'
        return table

    def test_new_replaces_existing(self):
        existing = self._make_table([100], ['b0', 'r0'])
        new = self._make_table([100], ['b0'])  # replace b0 row
        merged = merge_frames(existing, new)

        keys = ['{}-{}'.format(e, c) for e, c in zip(merged['EXPID'], merged['CAMERA'])]
        self.assertIn('100-b0', keys)
        self.assertIn('100-r0', keys)

    def test_preserves_non_overlapping(self):
        existing = self._make_table([100], ['b0'])
        new = self._make_table([200], ['b0'])
        merged = merge_frames(existing, new)

        self.assertIn(100, merged['EXPID'].tolist())
        self.assertIn(200, merged['EXPID'].tolist())

    def test_new_column_in_new_table_added_to_result(self):
        """A column present only in new_table should appear in the merged result."""
        existing = self._make_table([100], ['b0'])
        new = self._make_table([200], ['b0'])
        new['EXTRA_COL'] = np.zeros(len(new), dtype=np.float32)
        merged = merge_frames(existing, new)
        self.assertIn('EXTRA_COL', merged.colnames)
        mask = merged['EXPID'] == 100
        self.assertAlmostEqual(float(merged['EXTRA_COL'][mask][0]), 0.0)


class TestWriteOutput(unittest.TestCase):
    """Test write_output() FITS and CSV writing."""

    def setUp(self):
        patcher = mock.patch('desispec.scripts.tsnr_afterburner.tsnr2_to_efftime',
                             side_effect=_mock_tsnr2_to_efftime)
        self.mock_efftime = patcher.start()
        self.addCleanup(patcher.stop)

    def _make_tables(self):
        row = _make_exposure_row(expid=100)
        camword_map = {100: ['b0', 'r0', 'z0']}
        return build_tables([row], camword_map=camword_map)

    def test_writes_fits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, 'test_out.fits')
            frames, exposures = self._make_tables()
            write_output(exposures, frames, outfile)
            self.assertTrue(os.path.isfile(outfile))

    def test_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, 'test_out.fits')
            frames, exposures = self._make_tables()
            write_output(exposures, frames, outfile)
            csv_file = outfile.replace('.fits', '.csv')
            self.assertTrue(os.path.isfile(csv_file))

    def test_fits_has_both_hdus(self):
        import astropy.io.fits as afits
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, 'test_out.fits')
            frames, exposures = self._make_tables()
            write_output(exposures, frames, outfile)
            hdus = afits.open(outfile)
            extnames = [h.name for h in hdus]
            self.assertIn('EXPOSURES', extnames)
            self.assertIn('FRAMES', extnames)
            hdus.close()

    def test_atomic_write_no_partial_file_on_failure(self):
        """If writing fails, the original file should not be corrupted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, 'test_out.fits')
            frames, exposures = self._make_tables()
            # Write once so the file exists
            write_output(exposures, frames, outfile)
            # Content size before
            size_before = os.path.getsize(outfile)
            # Write again (overwrite=True for tmpfile)
            write_output(exposures, frames, outfile)
            self.assertTrue(os.path.isfile(outfile))

    def test_csv_rounding_precision(self):
        """TIME/SNR2 columns rounded to 1 dp; SEEING/MAG/etc columns rounded to 3 dp."""
        import astropy.table
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, 'test_out.fits')
            frames, exposures = self._make_tables()
            # Force values that would expose rounding
            for col in exposures.colnames:
                if any(x in col for x in ('TIME', 'SNR2')):
                    try:
                        exposures[col] = exposures[col].astype(float) + 0.123456
                    except Exception:
                        pass
                elif any(x in col for x in ('SEEING', 'EBV', 'AIRMASS')):
                    try:
                        exposures[col] = exposures[col].astype(float) + 0.123456
                    except Exception:
                        pass
            write_output(exposures, frames, outfile)
            csv_file = outfile.replace('.fits', '.csv')
            csv = astropy.table.Table.read(csv_file)

            for col in csv.colnames:
                if any(x in col for x in ('TIME', 'SNR2')):
                    val = float(csv[col][0])
                    self.assertAlmostEqual(val, round(val, 1), places=6,
                                           msg='Column {} not rounded to 1 dp'.format(col))
                elif any(x in col for x in ('SEEING', 'EBV', 'AIRMASS')):
                    val = float(csv[col][0])
                    self.assertAlmostEqual(val, round(val, 3), places=8,
                                           msg='Column {} not rounded to 3 dp'.format(col))


class TestPetalqaCameraValues(unittest.TestCase):
    """Test _petalqa_camera_values() handling of PETALQA placeholder zeros."""

    def test_recorded_band(self):
        values = _petalqa_camera_values(_make_petalqa([0, 1]), 'r1')
        self.assertEqual(len(values), len(_TSNR2_TRACERS))
        self.assertEqual(float(values['TSNR2_ELG']), 10.0)

    def test_single_zero_tracer_is_kept(self):
        petalqa = _make_petalqa([0])
        petalqa['TSNR2_LYA_Z'] = 0.0
        values = _petalqa_camera_values(petalqa, 'z0')
        self.assertEqual(float(values['TSNR2_LYA']), 0.0)
        self.assertEqual(len(values), len(_TSNR2_TRACERS))

    def test_all_zero_band_is_missing(self):
        petalqa = _make_petalqa([0])
        for tracer in _TSNR2_TRACERS:
            petalqa['TSNR2_{}_B'.format(tracer)] = 0.0
        self.assertEqual(_petalqa_camera_values(petalqa, 'b0'), {})
        self.assertEqual(len(_petalqa_camera_values(petalqa, 'r0')), len(_TSNR2_TRACERS))

    def test_missing_petal_or_qa(self):
        self.assertEqual(_petalqa_camera_values(_make_petalqa([0]), 'b5'), {})
        self.assertEqual(_petalqa_camera_values(None, 'b0'), {})

    def test_nan_value_omitted(self):
        petalqa = _make_petalqa([0])
        petalqa['TSNR2_ELG_B'] = np.nan
        values = _petalqa_camera_values(petalqa, 'b0')
        self.assertNotIn('TSNR2_ELG', values)
        self.assertEqual(len(values), len(_TSNR2_TRACERS) - 1)


class TestFillTargFromHeader(unittest.TestCase):
    """Test _fill_targ_from_header() precedence rules."""

    def test_fills_defaults_only(self):
        entry = dict(_TARG_DEFAULTS)
        entry['SURVEY'] = 'sv3'
        hdr = {'SURVEY': 'main', 'FAFLAVOR': 'MAINDARK', 'MINTFRAC': 0.85, 'GOALTYPE': 'DARK'}
        entry = _fill_targ_from_header(entry, hdr)
        self.assertEqual(entry['SURVEY'], 'sv3')
        self.assertEqual(entry['FAFLAVOR'], 'maindark')
        self.assertEqual(entry['GOALTYPE'], 'dark')
        self.assertEqual(entry['MINTFRAC'], 0.85)
        self.assertEqual(entry['FAPRGRM'], 'unknown')

    def test_goaltime_sentinel_replaced(self):
        """Exposure-table GOALTIME=-99 is treated as missing."""
        entry = dict(_TARG_DEFAULTS)
        entry['GOALTIME'] = -99.
        self.assertEqual(_fill_targ_from_header(entry, {'GOALTIME': 180.})['GOALTIME'], 180.)
        entry['GOALTIME'] = 300.
        self.assertEqual(_fill_targ_from_header(entry, {'GOALTIME': 180.})['GOALTIME'], 300.)

    def test_fa_surv_used_for_survey(self):
        entry = _fill_targ_from_header(dict(_TARG_DEFAULTS), {'FA_SURV': 'Main'})
        self.assertEqual(entry['SURVEY'], 'main')


class TestFindProcessedNights(unittest.TestCase):
    """Test find_processed_nights() uses the exposures directory."""

    def test_nights_from_exposures_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ('20210628', '20220211', 'attic', '2021'):
                os.makedirs(os.path.join(tmpdir, 'exposures', name))
            with patch('desispec.scripts.tsnr_afterburner.specprod_root', return_value=tmpdir):
                self.assertEqual(find_processed_nights(), [20210628, 20220211])


class TestAddEfftimes(unittest.TestCase):
    """Direct tests for _add_efftimes helper."""

    def setUp(self):
        patcher = mock.patch('desispec.scripts.tsnr_afterburner.tsnr2_to_efftime',
                             side_effect=_mock_tsnr2_to_efftime)
        self.mock_efftime = patcher.start()
        self.addCleanup(patcher.stop)

    def _base_row(self, night, goaltype, tsnr2_val=100.0):
        row = {'NIGHT': night, 'GOALTYPE': goaltype}
        for tracer in _TSNR2_TRACERS:
            row['TSNR2_{}'.format(tracer)] = np.float32(tsnr2_val)
        return row

    def test_dark_before_cutoff_uses_elg(self):
        row = self._base_row(20210801, 'dark')
        _add_efftimes(row)
        self.assertAlmostEqual(float(row['EFFTIME_SPEC']),
                                float(row['ELG_EFFTIME_DARK']), places=3)

    def test_dark_after_cutoff_uses_lrg(self):
        row = self._base_row(20211001, 'dark')
        _add_efftimes(row)
        self.assertAlmostEqual(float(row['EFFTIME_SPEC']),
                                float(row['LRG_EFFTIME_DARK']), places=3)

    def test_bright_uses_bgs(self):
        row = self._base_row(20211001, 'bright')
        _add_efftimes(row)
        self.assertAlmostEqual(float(row['EFFTIME_SPEC']),
                                float(row['BGS_EFFTIME_BRIGHT']), places=3)

    def test_backup_uses_gpb_backup(self):
        row = self._base_row(20211001, 'backup')
        _add_efftimes(row)
        self.assertAlmostEqual(float(row['EFFTIME_SPEC']),
                                float(row['GPB_EFFTIME_BACKUP']), places=3)

    def test_backup_fallback_uses_bgs_when_gpb_zero(self):
        """When GPB_EFFTIME_BACKUP is zero for a backup exposure, fall back to BGS."""
        row = self._base_row(20211001, 'backup', tsnr2_val=100.0)
        row['TSNR2_GPBBACKUP'] = np.float32(0.0)
        _add_efftimes(row)
        # GPB_EFFTIME_BACKUP will be 0 (0.0 * 0.1); EFFTIME_SPEC should fall back to BGS
        self.assertAlmostEqual(float(row['GPB_EFFTIME_BACKUP']), 0.0, places=5)
        self.assertAlmostEqual(float(row['EFFTIME_SPEC']),
                                float(row['BGS_EFFTIME_BRIGHT']), places=3)


# ---------------------------------------------------------------------------
# Phase 3a: TestReadEtcValues
# ---------------------------------------------------------------------------

class TestReadEtcValues(unittest.TestCase):
    """Direct tests for _read_etc_values() header/JSON fallback logic."""

    def test_both_keys_in_header(self):
        """When ACQFWHM and ETCTEFF are in the header, no filesystem access occurs."""
        hdr = {'ACQFWHM': 1.5, 'ETCTEFF': 800.0}
        with patch('desispec.scripts.tsnr_afterburner.findfile') as mock_ff:
            seeing, efftime = _read_etc_values(20210601, 12345, hdr)
        mock_ff.assert_not_called()
        self.assertAlmostEqual(float(seeing), 1.5, places=5)
        self.assertAlmostEqual(float(efftime), 800.0, places=3)

    def test_acqfwhm_from_json_when_absent(self):
        """When ACQFWHM is absent from header, value is read from ETC JSON file."""
        hdr = {'ETCTEFF': 700.0}  # no ACQFWHM
        etc_data = {'expinfo': {'acq_fwhm': 1.3, 'efftime': 700.0}}
        with tempfile.TemporaryDirectory() as tmpdir:
            etc_file = os.path.join(tmpdir, 'etc.json')
            import json as _json
            with open(etc_file, 'w') as f:
                _json.dump(etc_data, f)
            with patch('desispec.scripts.tsnr_afterburner.findfile', return_value=etc_file):
                seeing, efftime = _read_etc_values(20210601, 12345, hdr)
        self.assertAlmostEqual(float(seeing), 1.3, places=5)

    def test_nan_efftime_clamped_to_zero(self):
        """NaN in ETCTEFF should be clamped to 0.0."""
        hdr = {'ACQFWHM': 1.2, 'ETCTEFF': float('nan')}
        with patch('desispec.scripts.tsnr_afterburner.findfile'):
            seeing, efftime = _read_etc_values(20210601, 12345, hdr)
        self.assertAlmostEqual(float(efftime), 0.0, places=5)

    def test_json_missing_expinfo_key_returns_zero_seeing(self):
        """JSON file without 'expinfo' should fall back to zeros for missing values."""
        hdr = {}  # no header keys at all
        etc_data = {'other_key': 42}
        with tempfile.TemporaryDirectory() as tmpdir:
            etc_file = os.path.join(tmpdir, 'etc.json')
            import json as _json
            with open(etc_file, 'w') as f:
                _json.dump(etc_data, f)
            with patch('desispec.scripts.tsnr_afterburner.findfile', return_value=etc_file):
                seeing, efftime = _read_etc_values(20210601, 12345, hdr)
        self.assertAlmostEqual(float(seeing), 0.0, places=5)
        self.assertAlmostEqual(float(efftime), 0.0, places=5)

    def test_no_json_file_returns_zeros(self):
        """When the ETC JSON file does not exist, returns (0.0, 0.0)."""
        hdr = {}
        with patch('desispec.scripts.tsnr_afterburner.findfile', return_value='/nonexistent/etc.json'):
            seeing, efftime = _read_etc_values(20210601, 12345, hdr)
        self.assertAlmostEqual(float(seeing), 0.0, places=5)
        self.assertAlmostEqual(float(efftime), 0.0, places=5)


# ---------------------------------------------------------------------------
# Phase 3b: TestCollectScienceExpids
# ---------------------------------------------------------------------------

class TestCollectScienceExpids(unittest.TestCase):
    """Tests for collect_science_expids() exposure table parsing."""

    def _make_exptab(self, rows):
        """Build a minimal astropy Table mimicking an exposure table.

        Args:
            rows: list of dict, one per exposure.  Keys: EXPID, TILEID, LASTSTEP,
                CAMWORD, BADCAMWORD, EXPTIME, MJD-OBS, EFFTIME_ETC, AIRMASS,
                SURVEY, FAPRGRM, FAFLAVOR, GOALTYPE, GOALTIME, EBVFAC, MINTFRAC.

        Returns:
            astropy.table.Table
        """
        defaults = {
            'CAMWORD': 'a0123456789', 'BADCAMWORD': '', 'EXPTIME': 900.0,
            'MJD-OBS': 59366.5, 'EFFTIME_ETC': 0.0, 'AIRMASS': 1.1,
            'SURVEY': 'main', 'FAPRGRM': 'dark', 'FAFLAVOR': 'maindark',
            'GOALTYPE': 'dark', 'GOALTIME': 1000.0, 'EBVFAC': 1.0,
            'MINTFRAC': 0.9,
        }
        filled = [{**defaults, **r} for r in rows]
        return Table(rows=filled)

    def _run(self, exptab_rows, nights=None, expids=None):
        """Patch findfile + load_table then call collect_science_expids."""
        from desispec.scripts.tsnr_afterburner import collect_science_expids
        exptab = self._make_exptab(exptab_rows)
        with patch('desispec.scripts.tsnr_afterburner.findfile', return_value='/fake/exptab.csv'), \
             patch('os.path.isfile', return_value=True), \
             patch('desispec.scripts.tsnr_afterburner.load_table', return_value=exptab):
            return collect_science_expids(nights=nights or [20210601], expids=expids)

    def test_good_exposure_laststep_all(self):
        """LASTSTEP='all' with valid TILEID goes to good_expids."""
        good, bad = self._run([{'EXPID': 1, 'TILEID': 1234, 'LASTSTEP': 'all'}])
        self.assertEqual(len(good), 1)
        self.assertEqual(len(bad), 0)
        self.assertEqual(good[0]['EXPID'], 1)

    def test_all_cameras_bad_goes_to_bad(self):
        """LASTSTEP='all' with CAMWORD == BADCAMWORD goes to bad_expids."""
        good, bad = self._run([{'EXPID': 3, 'TILEID': 1234, 'LASTSTEP': 'all',
                                'CAMWORD': 'a0123456789', 'BADCAMWORD': 'a0123456789'}])
        self.assertEqual(len(good), 0)
        self.assertEqual([b['EXPID'] for b in bad], [3])

    def test_bad_exposure_laststep_skysub(self):
        """LASTSTEP != 'all' with valid TILEID goes to bad_expids."""
        good, bad = self._run([{'EXPID': 2, 'TILEID': 5678, 'LASTSTEP': 'skysub'}])
        self.assertEqual(len(good), 0)
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0]['EXPID'], 2)

    def test_tileid_zero_skipped(self):
        """TILEID <= 0 (calibration exposure) is excluded from both lists."""
        good, bad = self._run([{'EXPID': 3, 'TILEID': 0, 'LASTSTEP': 'all'}])
        self.assertEqual(len(good), 0)
        self.assertEqual(len(bad), 0)

    def test_expids_filter_applied(self):
        """Only EXPIDs in the filter list should appear in output."""
        rows = [
            {'EXPID': 10, 'TILEID': 111, 'LASTSTEP': 'all'},
            {'EXPID': 20, 'TILEID': 222, 'LASTSTEP': 'all'},
        ]
        good, bad = self._run(rows, expids=[10])
        expids_out = [e['EXPID'] for e in good]
        self.assertIn(10, expids_out)
        self.assertNotIn(20, expids_out)

    def test_missing_exposure_table_skipped(self):
        """When the exposure table file does not exist, both lists are empty."""
        from desispec.scripts.tsnr_afterburner import collect_science_expids
        with patch('desispec.scripts.tsnr_afterburner.findfile', return_value='/nonexistent.csv'), \
             patch('os.path.isfile', return_value=False):
            good, bad = collect_science_expids(nights=[20210601])
        self.assertEqual(len(good), 0)
        self.assertEqual(len(bad), 0)


# ---------------------------------------------------------------------------
# Phase 3c: TestReadOneCamera
# ---------------------------------------------------------------------------

class TestReadOneCamera(unittest.TestCase):
    """Tests for read_one_camera() — the --recompute I/O path."""

    def _make_tsnr2_results(self, band='b', nfiber=500, value=10.0):
        """Return a (results_dict, alpha) pair mimicking calc_tsnr2 output."""
        results = {}
        band_upper = band.upper()
        for tracer in _TSNR2_TRACERS:
            results['TSNR2_{}_{}'.format(tracer, band_upper)] = np.full(nfiber, value, dtype=np.float32)
        return results, 1.0

    def _make_fibermap(self, nfiber=500):
        dt = [('EBV', np.float32), ('FIBER', np.int32)]
        fm = np.zeros(nfiber, dtype=dt)
        fm['EBV'] = 0.05
        return fm

    def _patch_all(self, flavor='science', flat_exists=True, cframe_exists=True):
        """Return a context manager stack that mocks all file I/O."""
        hdr0 = {
            'FLAVOR': flavor, 'TILEID': 1234, 'TILERA': 180.0, 'TILEDEC': 30.0,
            'MJD-OBS': 59366.5, 'EXPTIME': 900.0, 'AIRMASS': 1.1,
            'FIBERFLT': '/fake/flat.fits',
            'ACQFWHM': 1.2, 'ETCTEFF': 800.0,
            'SURVEY': 'main', 'FAPRGRM': 'dark', 'FAFLAVOR': 'maindark',
            'GOALTYPE': 'dark', 'GOALTIME': 1000.0, 'MINTFRAC': 0.9,
        }
        hdr1 = {}
        fibermap_hdr = {
            'SURVEY': 'main', 'FAPRGRM': 'dark', 'FAFLAVOR': 'maindark',
            'GOALTYPE': 'dark', 'GOALTIME': 1000.0, 'MINTFRAC': 0.9,
        }
        fibermap = self._make_fibermap()
        tsnr2_results, alpha = self._make_tsnr2_results(band='b')

        # Build per-extension mocks so that __getitem__ dispatches correctly.
        # MagicMock's default __getitem__ returns the same child for every key,
        # so we replace it with a side_effect that selects by key.
        mock_ext0 = MagicMock()
        mock_ext0.read_header.return_value = hdr0
        mock_ext1 = MagicMock()
        mock_ext1.read_header.return_value = hdr1
        mock_fibermap_ext = MagicMock()
        mock_fibermap_ext.read.return_value = fibermap
        mock_fibermap_ext.read_header.return_value = fibermap_hdr

        def _getitem(_self, key):
            if key == 0:
                return mock_ext0
            if key == 1:
                return mock_ext1
            if key == 'FIBERMAP':
                return mock_fibermap_ext
            raise KeyError('Unexpected FITS extension in test mock: {}'.format(key))

        mock_cframe_fits = MagicMock()
        mock_cframe_fits.__getitem__ = _getitem
        mock_cframe_fits.close = MagicMock()

        patches = [
            patch('desispec.scripts.tsnr_afterburner.findfile', return_value='/fake/file'),
            patch('desispec.scripts.tsnr_afterburner.os.path.isfile', return_value=cframe_exists),
            patch('desispec.scripts.tsnr_afterburner.os.path.exists', return_value=flat_exists),
            patch('desispec.scripts.tsnr_afterburner.fitsio.FITS', return_value=mock_cframe_fits),
            patch('desispec.scripts.tsnr_afterburner.read_frame', return_value=MagicMock()),
            patch('desispec.scripts.tsnr_afterburner.read_fiberflat', return_value=MagicMock()),
            patch('desispec.scripts.tsnr_afterburner.read_flux_calibration', return_value=MagicMock()),
            patch('desispec.scripts.tsnr_afterburner.read_sky', return_value=MagicMock()),
            patch('desispec.scripts.tsnr_afterburner.calc_tsnr2', return_value=(tsnr2_results, alpha)),
            patch('desispec.scripts.tsnr_afterburner.get_skymag_values',
                  return_value={'SKY_MAG_G_SPEC': 22.0, 'SKY_MAG_R_SPEC': 21.0, 'SKY_MAG_Z_SPEC': 19.0}),
        ]
        return patches

    def test_missing_cframe_returns_none(self):
        """If the cframe file does not exist, read_one_camera should return None."""
        from desispec.scripts.tsnr_afterburner import read_one_camera
        with patch('desispec.scripts.tsnr_afterburner.findfile', return_value='/fake/cframe'), \
             patch('desispec.scripts.tsnr_afterburner.os.path.isfile', return_value=False):
            result = read_one_camera(20210601, 12345, 'b0')
        self.assertIsNone(result)

    def test_non_science_flavor_returns_none(self):
        """Non-science FLAVOR (e.g. 'arc') should cause read_one_camera to return None."""
        from desispec.scripts.tsnr_afterburner import read_one_camera
        patches = self._patch_all(flavor='arc')
        with patches[0], patches[1], patches[2], patches[3]:
            result = read_one_camera(20210601, 12345, 'b0')
        self.assertIsNone(result)

    def test_flat_not_found_raises(self):
        """Missing calculation inputs must not silently replace real values."""
        from desispec.scripts.tsnr_afterburner import read_one_camera
        patches = self._patch_all(flavor='science', flat_exists=False)
        with patches[0], patches[1], patches[2], patches[3]:
            with self.assertRaises(FileNotFoundError):
                read_one_camera(20210601, 12345, 'b0')

    def test_returns_expected_keys(self):
        """Successful read_one_camera should return a dict with all required keys."""
        from desispec.scripts.tsnr_afterburner import read_one_camera
        patches = self._patch_all()
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9]:
            result = read_one_camera(20210601, 12345, 'b0')
        self.assertIsNotNone(result)
        required = ('NIGHT', 'EXPID', 'TILEID', 'CAMERA', 'TSNR2_ELG',
                    'TSNR2_LRG', 'TSNR2_BGS', 'SKY_MAG_G_SPEC', 'EBV')
        for key in required:
            self.assertIn(key, result, 'Missing key: {}'.format(key))
        self.assertEqual(result['CAMERA'], 'b0')

    def test_tsnr2_median_ignores_zeros(self):
        """Median TSNR2 should be computed only over nonzero fibers."""
        from desispec.scripts.tsnr_afterburner import read_one_camera
        nfiber = 500
        vals = np.zeros(nfiber, dtype=np.float32)
        vals[:250] = 20.0  # half nonzero
        tsnr2_results = {'TSNR2_ELG_B': vals}
        for tracer in _TSNR2_TRACERS:
            if tracer != 'ELG':
                tsnr2_results['TSNR2_{}_B'.format(tracer)] = vals.copy()
        patches = self._patch_all()
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], \
             patch('desispec.scripts.tsnr_afterburner.calc_tsnr2',
                   return_value=(tsnr2_results, 1.0)), patches[9]:
            result = read_one_camera(20210601, 12345, 'b0')
        self.assertIsNotNone(result)
        # median of nonzero values only = 20.0, not median(all) = 10.0
        self.assertAlmostEqual(float(result['TSNR2_ELG']), 20.0, places=2)


# ---------------------------------------------------------------------------
# Phase 3d: TestMain
# ---------------------------------------------------------------------------

class TestMain(unittest.TestCase):
    """Smoke tests for the main() entry point."""

    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        self._efftime_patcher = patch(
            'desispec.scripts.tsnr_afterburner.tsnr2_to_efftime',
            side_effect=_mock_tsnr2_to_efftime)
        self._efftime_patcher.start()
        self.addCleanup(self._efftime_patcher.stop)

    def _good_entry(self, night=20211001, expid=100):
        return {
            'NIGHT': night, 'EXPID': expid, 'TILEID': 1234,
            'CAMWORD': 'a0123456789', 'BADCAMWORD': '',
            'EXPTIME': 900.0, 'MJD-OBS': 59366.5, 'EFFTIME_ETC': 800.0,
            'AIRMASS': 1.1, 'SURVEY': 'main', 'FAPRGRM': 'dark',
            'FAFLAVOR': 'maindark', 'GOALTYPE': 'dark',
            'GOALTIME': 1000.0, 'MINTFRAC': 0.9, 'EBVFAC': 1.0,
            'LASTSTEP': 'all',
        }

    def _run_main(self, extra_argv=None, outfile=None, tmpdir=None):
        """Call main() with minimal patching for a non-MPI, non-recompute, no-GFA run."""
        from desispec.scripts.tsnr_afterburner import main
        exposure_row = _make_exposure_row(expid=100, night=20211001)

        argv = ['--prod', '/fake/prod', '--nights', '20211001']
        if outfile:
            argv += ['-o', outfile]
        if extra_argv:
            argv += extra_argv

        with patch('desispec.scripts.tsnr_afterburner.specprod_root', return_value='/fake/prod'), \
             patch('desispec.scripts.tsnr_afterburner.collect_science_expids',
                   return_value=([self._good_entry()], [])), \
             patch('desispec.scripts.tsnr_afterburner.read_one_exposure',
                   return_value=exposure_row), \
             patch('os.path.isfile', return_value=False), \
             patch('desispec.scripts.tsnr_afterburner.write_output') as mock_write:
            rc = main(argv)
        return rc, mock_write

    def test_main_returns_zero_on_success(self):
        """main() should return 0 on a clean run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, 'out.fits')
            rc, _ = self._run_main(outfile=outfile)
        self.assertEqual(rc, 0)

    def test_non_fits_extension_returns_one(self):
        """main() with a .txt outfile that cannot be corrected should return 1."""
        from desispec.scripts.tsnr_afterburner import main
        with patch('desispec.scripts.tsnr_afterburner.specprod_root', return_value='/fake/prod'):
            rc = main(['--prod', '/fake/prod', '-o', '/tmp/out.txt', '--nights', '20211001'])
        self.assertEqual(rc, 1)

    def test_csv_outfile_corrected_to_fits(self):
        """main() with a .csv outfile should auto-correct to .fits and return 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_out = os.path.join(tmpdir, 'out.csv')
            rc, mock_write = self._run_main(outfile=csv_out)
        self.assertEqual(rc, 0)
        # write_output should have been called with a .fits path
        actual_outfile = mock_write.call_args[0][2]
        self.assertTrue(actual_outfile.endswith('.fits'),
                        'write_output called with non-.fits path: {}'.format(actual_outfile))

    def test_overwrite_skips_preexisting_merge(self):
        """--overwrite should cause main() to skip reading any pre-existing output file."""
        from desispec.scripts.tsnr_afterburner import main
        exposure_row = _make_exposure_row(expid=100, night=20211001)
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, 'out.fits')
            with patch('desispec.scripts.tsnr_afterburner.specprod_root', return_value='/fake/prod'), \
                 patch('desispec.scripts.tsnr_afterburner.collect_science_expids',
                       return_value=([self._good_entry()], [])), \
                 patch('desispec.scripts.tsnr_afterburner.read_one_exposure',
                       return_value=exposure_row), \
                 patch('desispec.scripts.tsnr_afterburner.write_output'), \
                 patch('desispec.scripts.tsnr_afterburner.read_table') as mock_read_table, \
                 patch('os.path.isfile', return_value=True):
                main(['--prod', '/fake/prod', '-o', outfile, '--nights', '20211001', '--overwrite'])
            mock_read_table.assert_not_called()

    def _run_main_with_existing(self, existing_nights, existing_expids, extra_argv):
        """Run main() against a mocked pre-existing output file; return (rc, write mock)."""
        from desispec.scripts.tsnr_afterburner import main
        exposure_row = _make_exposure_row(expid=100, night=20211001)
        existing = Table({'NIGHT': np.array(existing_nights, dtype=np.int32),
                          'EXPID': np.array(existing_expids, dtype=np.int32)})
        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, 'out.fits')
            with patch('desispec.scripts.tsnr_afterburner.specprod_root', return_value='/fake/prod'), \
                 patch('desispec.scripts.tsnr_afterburner.collect_science_expids',
                       return_value=([self._good_entry()], [])) as mock_collect, \
                 patch('desispec.scripts.tsnr_afterburner.read_one_exposure',
                       return_value=exposure_row), \
                 patch('desispec.scripts.tsnr_afterburner.read_table', return_value=existing), \
                 patch('desispec.scripts.tsnr_afterburner.merge_exposures',
                       side_effect=lambda old, new: new), \
                 patch('desispec.scripts.tsnr_afterburner.merge_frames',
                       side_effect=lambda old, new, **kwargs: new), \
                 patch('desispec.scripts.tsnr_afterburner.write_output') as mock_write, \
                 patch('os.path.isfile', return_value=True):
                rc = main(['--prod', '/fake/prod', '-o', outfile, '--nights', '20211001'] + extra_argv)
        return rc, mock_write, mock_collect

    def test_no_update_errors_when_night_present(self):
        """--no-update should exit with 1 before processing if a requested night is in the file."""
        rc, mock_write, mock_collect = self._run_main_with_existing(
            [20211001], [99], ['--no-update'])
        self.assertEqual(rc, 1)
        mock_collect.assert_not_called()
        mock_write.assert_not_called()

    def test_no_update_adds_when_night_absent(self):
        """--no-update should proceed and write when no requested night is in the file."""
        rc, mock_write, _ = self._run_main_with_existing([20210930], [99], ['--no-update'])
        self.assertEqual(rc, 0)
        mock_write.assert_called_once()

    def test_no_update_with_expids_checks_only_those_expids(self):
        """With --expids, --no-update conflicts only if those exposures are already present."""
        rc, _, _ = self._run_main_with_existing([20211001], [99], ['--no-update', '--expids', '100'])
        self.assertEqual(rc, 0)
        rc, _, _ = self._run_main_with_existing([20211001], [100], ['--no-update', '--expids', '100'])
        self.assertEqual(rc, 1)

    def test_default_updates_when_night_present(self):
        """Without --no-update, rows for a night already in the file are replaced."""
        rc, mock_write, _ = self._run_main_with_existing([20211001], [100], [])
        self.assertEqual(rc, 0)
        mock_write.assert_called_once()

    def test_add_badexp_calls_inject(self):
        """--add-badexp with a non-empty bad_expids list should call inject_bad_exposures."""
        from desispec.scripts.tsnr_afterburner import main
        exposure_row = _make_exposure_row(expid=100, night=20211001)
        bad_entry = self._good_entry(expid=999)
        bad_entry['LASTSTEP'] = 'skysub'

        # The mock must return non-empty tables; returning empty tables would cause
        # main() to hit the "No valid exposures" guard and return 1 before write_output.
        def _inject_passthrough(exp_tbl, frm_tbl, bad_list, cameras=None):
            return exp_tbl, frm_tbl

        with tempfile.TemporaryDirectory() as tmpdir:
            outfile = os.path.join(tmpdir, 'out.fits')
            with patch('desispec.scripts.tsnr_afterburner.specprod_root', return_value='/fake/prod'), \
                 patch('desispec.scripts.tsnr_afterburner.collect_science_expids',
                       return_value=([self._good_entry()], [bad_entry])), \
                 patch('desispec.scripts.tsnr_afterburner.read_one_exposure',
                       return_value=exposure_row), \
                 patch('desispec.scripts.tsnr_afterburner.write_output'), \
                 patch('os.path.isfile', return_value=False), \
                 patch('desispec.scripts.tsnr_afterburner.inject_bad_exposures',
                       side_effect=_inject_passthrough) as mock_inject:
                rc = main(['--prod', '/fake/prod', '-o', outfile,
                           '--nights', '20211001', '--add-badexp'])
            mock_inject.assert_called_once()
            self.assertEqual(rc, 0)


if __name__ == '__main__':
    unittest.main()
