"""
Tests for desispec.inventory

Groups:
  TestInventoryPure      - pure functions, no I/O, run everywhere
  TestInventorySynthetic - HDF5 creation and query using synthetic data, run everywhere
  TestInventoryHealpix   - NERSC-only, healpix-based specprod (loa)
  TestInventoryUniqpix   - NERSC-only, uniqpix-based specprod (matterhorn)
"""

import os
import glob
import shutil
import tempfile
import unittest

import numpy as np
import h5py
from astropy.table import Table

from desispec.inventory import (
    _get_unique_indices,
    _create_header,
    create_inventory_zcat,
    write_inventory,
    target_tiles,
    target_healpix,
    radec2targetids,
    parse_radec,
    update_inventory,
    _get_default_inventory_filename,
)

# ---------------------------------------------------------------------------
# NERSC availability checks - wrapped so the import doesn't fail off-NERSC
# ---------------------------------------------------------------------------

_HEALPIX_SPECPROD = 'loa'        # healpix-based; update when loa is retired
_UNIQPIX_SPECPROD = 'matterhorn' # uniqpix-based; update when matterhorn is retired

def _try_specprod_root(specprod):
    try:
        import desispec.io
        return desispec.io.specprod_root(specprod, readonly=True)
    except (KeyError, Exception):
        return None

_healpix_root = _try_specprod_root(_HEALPIX_SPECPROD)
_uniqpix_root = _try_specprod_root(_UNIQPIX_SPECPROD)

at_nersc_healpix = ('NERSC_HOST' in os.environ) and (_healpix_root is not None) and os.path.exists(_healpix_root)
at_nersc_uniqpix = ('NERSC_HOST' in os.environ) and (_uniqpix_root is not None) and os.path.exists(_uniqpix_root)


# ---------------------------------------------------------------------------
# Helper: build a minimal synthetic zcat Table with correct healpix pixels
# ---------------------------------------------------------------------------

def _make_synthetic_zcat(ngroups=10):
    """Return a small zcat Table suitable for create_inventory_zcat testing.

    HEALPIX values are computed from TARGET_RA/DEC at nside=64 so that
    radec2targetids (which uses nside=64) can find them.
    """
    import healpy
    ra  = np.array([10.0, 10.01, 10.02, 20.0, 20.01])
    dec = np.array([0.0,  0.01,  0.02,  5.0,  5.01])
    zcat = Table()
    zcat['TARGETID']   = np.array([1001, 1002, 1003, 2001, 2002], dtype=np.int64)
    zcat['TILEID']     = np.array([100, 100, 100, 200, 200], dtype=np.int32)
    zcat['LASTNIGHT']  = np.array([20230101]*5, dtype=np.int32)
    zcat['FIBER']      = np.array([0, 1, 2, 0, 1], dtype=np.int32)
    zcat['TARGET_RA']  = ra
    zcat['TARGET_DEC'] = dec
    zcat['SURVEY']     = np.array(['main']*5)
    zcat['PROGRAM']    = np.array(['dark']*5)
    zcat['HEALPIX']    = healpy.ang2pix(64, ra, dec, lonlat=True, nest=True).astype(np.int32)
    return zcat


# ===========================================================================
# Group 1: Pure functions
# ===========================================================================

class TestInventoryPure(unittest.TestCase):

    def test_get_unique_indices_basic(self):
        arr = np.array([3, 1, 2, 1, 3, 3])
        result = _get_unique_indices(arr)
        self.assertEqual(set(result.keys()), {1, 2, 3})
        np.testing.assert_array_equal(sorted(result[1]), [1, 3])
        np.testing.assert_array_equal(sorted(result[2]), [2])
        np.testing.assert_array_equal(sorted(result[3]), [0, 4, 5])

    def test_get_unique_indices_single_value(self):
        arr = np.array([7, 7, 7])
        result = _get_unique_indices(arr)
        self.assertEqual(list(result.keys()), [7])
        np.testing.assert_array_equal(sorted(result[7]), [0, 1, 2])

    def test_get_unique_indices_all_unique(self):
        arr = np.array([10, 20, 30])
        result = _get_unique_indices(arr)
        for val in [10, 20, 30]:
            self.assertEqual(len(result[val]), 1)

    def test_parse_radec_string_two(self):
        ra, dec, radius = parse_radec('12.5,34.7')
        self.assertAlmostEqual(ra, 12.5)
        self.assertAlmostEqual(dec, 34.7)
        self.assertAlmostEqual(radius, 10.0)

    def test_parse_radec_string_three(self):
        ra, dec, radius = parse_radec('12.5,34.7,30.0')
        self.assertAlmostEqual(ra, 12.5)
        self.assertAlmostEqual(dec, 34.7)
        self.assertAlmostEqual(radius, 30.0)

    def test_parse_radec_list(self):
        ra, dec, radius = parse_radec([1.0, 2.0])
        self.assertAlmostEqual(ra, 1.0)
        self.assertAlmostEqual(dec, 2.0)
        self.assertAlmostEqual(radius, 10.0)

    def test_parse_radec_tuple_three(self):
        ra, dec, radius = parse_radec((5.0, 6.0, 20.0))
        self.assertAlmostEqual(radius, 20.0)

    def test_parse_radec_bad_length(self):
        with self.assertRaises(ValueError):
            parse_radec([1.0])
        with self.assertRaises(ValueError):
            parse_radec([1.0, 2.0, 3.0, 4.0])

    def test_create_header_full(self):
        hdr = _create_header((10.0, 20.0, 30.0), 'myprod')
        self.assertAlmostEqual(hdr['RA'], 10.0)
        self.assertAlmostEqual(hdr['DEC'], 20.0)
        self.assertAlmostEqual(hdr['RADIUS'], 30.0)
        self.assertEqual(hdr['SPECPROD'], 'myprod')

    def test_create_header_no_radec(self):
        hdr = _create_header(None, 'myprod')
        self.assertNotIn('RA', hdr)
        self.assertEqual(hdr['SPECPROD'], 'myprod')

    def test_create_header_specprod_env(self):
        old = os.environ.get('SPECPROD')
        os.environ['SPECPROD'] = 'testprod'
        try:
            hdr = _create_header(None, None)
            self.assertEqual(hdr['SPECPROD'], 'testprod')
        finally:
            if old is None:
                os.environ.pop('SPECPROD', None)
            else:
                os.environ['SPECPROD'] = old

    def test_update_inventory_raises(self):
        with self.assertRaises(NotImplementedError):
            update_inventory('dummy.h5')


# ===========================================================================
# Group 2: HDF5 creation and query with synthetic data
# ===========================================================================

class TestInventorySynthetic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.h5file = os.path.join(cls.tmpdir, 'test_inventory.h5')
        cls.ngroups = 10
        cls.zcat = _make_synthetic_zcat(cls.ngroups)
        create_inventory_zcat(cls.zcat, cls.h5file, ngroups=cls.ngroups)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)

    # --- write_inventory ---

    def test_write_inventory_creates_file(self):
        outfile = os.path.join(self.tmpdir, 'write_test.h5')
        inv = {
            'target_tiles': {0: Table({'TARGETID': np.array([1], dtype=np.int64),
                                       'TILEID': np.array([10], dtype=np.int32),
                                       'LASTNIGHT': np.array([20230101], dtype=np.int32),
                                       'FIBER': np.array([0], dtype=np.int32)})},
        }
        write_inventory(outfile, inv, ngroups=5)
        self.assertTrue(os.path.exists(outfile))
        with h5py.File(outfile) as hx:
            self.assertEqual(hx.attrs['ngroups'], 5)
            self.assertIn('target_tiles', hx)

    # --- create_inventory_zcat structure ---

    def test_create_inventory_zcat_healpix_groups(self):
        with h5py.File(self.h5file) as hx:
            self.assertIn('target_healpix', hx)
            self.assertIn('healpix_targets', hx)
            self.assertIn('target_tiles', hx)
            self.assertIn('tile_targets', hx)
            self.assertNotIn('target_uniqpix', hx)
            self.assertNotIn('uniqpix_targets', hx)

    def test_create_inventory_zcat_ngroups_attr(self):
        with h5py.File(self.h5file) as hx:
            self.assertEqual(hx.attrs['ngroups'], self.ngroups)

    def test_create_inventory_zcat_uniqpix_groups(self):
        outfile = os.path.join(self.tmpdir, 'upix.h5')
        zcat = self.zcat.copy()
        zcat.remove_column('HEALPIX')
        zcat['UNIQPIX'] = np.array([500, 500, 500, 600, 600], dtype=np.int32)
        create_inventory_zcat(zcat, outfile, ngroups=self.ngroups)
        with h5py.File(outfile) as hx:
            self.assertIn('target_uniqpix', hx)
            self.assertIn('uniqpix_targets', hx)
            self.assertNotIn('target_healpix', hx)
            self.assertNotIn('healpix_targets', hx)

    def test_create_inventory_zcat_hpix2upix(self):
        outfile = os.path.join(self.tmpdir, 'hpix2upix.h5')
        zcat = self.zcat.copy()
        zcat.remove_column('HEALPIX')
        nside = 512
        zcat['UNIQPIX'] = np.array([500, 500, 500, 600, 600], dtype=np.int32)
        zcat.meta['NSIDEMAX'] = nside
        npix = 12 * nside * nside
        h2u = np.arange(npix, dtype=np.int32)
        hpix2upix = {'main': {'dark': h2u}}
        create_inventory_zcat(zcat, outfile, ngroups=self.ngroups, hpix2upix=hpix2upix)
        with h5py.File(outfile) as hx:
            self.assertIn('hpix2upix', hx)
            self.assertIn('main', hx['hpix2upix'])
            self.assertIn('dark', hx['hpix2upix/main'])
            self.assertEqual(hx['hpix2upix'].attrs['nside'], nside)

    def test_create_inventory_zcat_no_pix_raises(self):
        outfile = os.path.join(self.tmpdir, 'nopix.h5')
        zcat = self.zcat.copy()
        zcat.remove_column('HEALPIX')
        with self.assertRaises(ValueError):
            create_inventory_zcat(zcat, outfile, ngroups=self.ngroups)

    # --- target_tiles ---

    def test_target_tiles_by_targetid(self):
        result = target_tiles(targetids=[1001, 1002], filename=self.h5file)
        self.assertIn('TARGETID', result.colnames)
        self.assertIn('TILEID', result.colnames)
        self.assertIn('LASTNIGHT', result.colnames)
        self.assertIn('FIBER', result.colnames)
        self.assertEqual(len(result), 2)
        self.assertIn(1001, result['TARGETID'])
        self.assertIn(1002, result['TARGETID'])

    def test_target_tiles_single_int(self):
        result = target_tiles(targetids=1001, filename=self.h5file)
        self.assertEqual(len(result), 1)
        self.assertEqual(result['TARGETID'][0], 1001)

    def test_target_tiles_missing_targetid(self):
        # 9991 % 10 == 1, so subgroup 1 exists but TARGETID 9991 is not in it
        result = target_tiles(targetids=[9991], filename=self.h5file)
        self.assertEqual(len(result), 0)
        self.assertIn('TARGETID', result.colnames)
        self.assertIn('TILEID', result.colnames)
        self.assertIn('LASTNIGHT', result.colnames)
        self.assertIn('FIBER', result.colnames)

    def test_target_tiles_inventory_dict(self):
        # pre-load inventory into memory and verify same results as file path
        with h5py.File(self.h5file) as hx:
            ngroups = hx.attrs['ngroups']
            inv = {'meta': {'ngroups': ngroups}, 'target_tiles': {}}
            for key in hx['target_tiles'].keys():
                inv['target_tiles'][int(key)] = hx[f'target_tiles/{key}'][:]

        result = target_tiles(targetids=[1001], filename=self.h5file, inventory=inv)
        self.assertEqual(len(result), 1)
        self.assertEqual(result['TARGETID'][0], 1001)

    def test_target_tiles_column_order(self):
        result = target_tiles(targetids=[1001], filename=self.h5file)
        self.assertEqual(list(result.colnames), ['TARGETID', 'TILEID', 'LASTNIGHT', 'FIBER'])

    # --- target_healpix ---

    def test_target_healpix_by_targetid(self):
        result = target_healpix(targetids=[1001, 2001], filename=self.h5file)
        self.assertIn('TARGETID', result.colnames)
        self.assertIn('SURVEY', result.colnames)
        self.assertIn('PROGRAM', result.colnames)
        self.assertIn('HEALPIX', result.colnames)
        self.assertEqual(len(result), 2)

    def test_target_healpix_values(self):
        result = target_healpix(targetids=[1001], filename=self.h5file)
        # SURVEY and PROGRAM are stored as bytes in HDF5
        survey = result['SURVEY'][0]
        if hasattr(survey, 'decode'):
            survey = survey.decode()
        self.assertEqual(survey, 'main')

    def test_target_healpix_missing_targetid(self):
        # 9991 % 10 == 1, so subgroup 1 exists but TARGETID 9991 is not in it
        result = target_healpix(targetids=[9991], filename=self.h5file)
        self.assertEqual(len(result), 0)
        self.assertIn('SURVEY', result.colnames)
        self.assertIn('PROGRAM', result.colnames)
        self.assertIn('HEALPIX', result.colnames)

    # --- radec2targetids ---

    def test_radec2targetids_finds_targets(self):
        # query near first cluster of synthetic targets (ra~10, dec~0)
        ra, dec = 10.01, 0.01
        targetids = radec2targetids((ra, dec, 120.0), filename=self.h5file)
        self.assertGreater(len(targetids), 0)
        # all three targets in that cluster should be found within 120 arcsec
        for tid in [1001, 1002, 1003]:
            self.assertIn(tid, targetids)

    def test_radec2targetids_excludes_far_targets(self):
        # query near first cluster; second cluster at ra~20 dec~5 should not appear
        ra, dec = 10.01, 0.01
        targetids = radec2targetids((ra, dec, 120.0), filename=self.h5file)
        for tid in [2001, 2002]:
            self.assertNotIn(tid, targetids)

    def test_radec2targetids_tiny_radius(self):
        # 0.001 arcsec won't contain any target
        targetids = radec2targetids((10.005, 0.005, 0.001), filename=self.h5file)
        self.assertEqual(len(targetids), 0)

    def test_radec2targetids_missing_pixel(self):
        # file with no healpix_targets entries should return empty array, not raise
        outfile = os.path.join(self.tmpdir, 'empty_hpix.h5')
        with h5py.File(outfile, 'w') as hx:
            hx.attrs['ngroups'] = 10
            hx.create_group('healpix_targets')
        targetids = radec2targetids((10.0, 0.0, 60.0), filename=outfile)
        self.assertEqual(len(targetids), 0)

    # --- target_tiles and target_healpix via radec ---

    def test_target_tiles_by_radec(self):
        result = target_tiles(radec=(10.01, 0.01, 120.0), filename=self.h5file)
        self.assertIn('TARGETID', result.colnames)
        self.assertGreater(len(result), 0)

    def test_target_healpix_by_radec(self):
        result = target_healpix(radec=(10.01, 0.01, 120.0), filename=self.h5file)
        self.assertIn('HEALPIX', result.colnames)
        self.assertGreater(len(result), 0)

    def test_target_tiles_meta_radec(self):
        radec = (10.01, 0.01, 120.0)
        result = target_tiles(radec=radec, filename=self.h5file)
        self.assertAlmostEqual(result.meta['RA'], 10.01)
        self.assertAlmostEqual(result.meta['DEC'], 0.01)


# ===========================================================================
# Group 3a: NERSC healpix-based specprod
# ===========================================================================

@unittest.skipUnless(at_nersc_healpix, f'not at NERSC or {_HEALPIX_SPECPROD} specprod missing')
class TestInventoryHealpix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import desispec.io
        from desispec.inventory import create_inventory
        cls.tmpdir = tempfile.mkdtemp()
        cls.h5file = os.path.join(cls.tmpdir, f'target_inventory-{_HEALPIX_SPECPROD}.h5')
        create_inventory(cls.h5file, specprod=_HEALPIX_SPECPROD, ntiles=2)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)

    def test_create_inventory_ntiles_structure(self):
        self.assertTrue(os.path.exists(self.h5file))
        with h5py.File(self.h5file) as hx:
            self.assertIn('ngroups', hx.attrs)
            self.assertIn('healpix_targets', hx)
            self.assertIn('target_healpix', hx)
            self.assertIn('target_tiles', hx)
            self.assertIn('tile_targets', hx)

    def test_inventory_tiledir(self):
        import desispec.io
        from desispec.inventory import _inventory_tiledir
        specdir = desispec.io.specprod_root(_HEALPIX_SPECPROD, readonly=True)
        tiledirs = sorted(glob.glob(os.path.join(specdir, 'tiles', 'cumulative', '[0-9]*')))
        self.assertGreater(len(tiledirs), 0)
        result = _inventory_tiledir(tiledirs[0])
        for col in ('TARGETID', 'TILEID', 'LASTNIGHT', 'FIBER', 'TARGET_RA', 'TARGET_DEC'):
            self.assertIn(col, result.colnames)
        self.assertGreater(len(result), 0)

    def test_nersc_target_tiles(self):
        # grab a real TARGETID from the inventory
        with h5py.File(self.h5file) as hx:
            first_group = next(iter(hx['target_tiles'].keys()))
            tid = int(hx[f'target_tiles/{first_group}']['TARGETID'][0])
        result = target_tiles(targetids=[tid], filename=self.h5file)
        self.assertEqual(len(result), 1)
        self.assertEqual(result['TARGETID'][0], tid)
        for col in ('TILEID', 'LASTNIGHT', 'FIBER'):
            self.assertIn(col, result.colnames)

    def test_nersc_target_healpix(self):
        with h5py.File(self.h5file) as hx:
            first_group = next(iter(hx['target_healpix'].keys()))
            tid = int(hx[f'target_healpix/{first_group}']['TARGETID'][0])
        result = target_healpix(targetids=[tid], filename=self.h5file)
        self.assertEqual(len(result), 1)
        self.assertEqual(result['TARGETID'][0], tid)
        for col in ('SURVEY', 'PROGRAM', 'HEALPIX'):
            self.assertIn(col, result.colnames)

    def test_get_default_inventory_filename(self):
        old_env = os.environ.get('DESI_TARGET_INVENTORY_DIR')
        os.environ['DESI_TARGET_INVENTORY_DIR'] = self.tmpdir
        try:
            found = _get_default_inventory_filename(_HEALPIX_SPECPROD)
            self.assertEqual(found, self.h5file)
        finally:
            if old_env is None:
                os.environ.pop('DESI_TARGET_INVENTORY_DIR', None)
            else:
                os.environ['DESI_TARGET_INVENTORY_DIR'] = old_env


# ===========================================================================
# Group 3b: NERSC uniqpix-based specprod
# ===========================================================================

@unittest.skipUnless(at_nersc_uniqpix, f'not at NERSC or {_UNIQPIX_SPECPROD} specprod missing')
class TestInventoryUniqpix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from desispec.inventory import create_inventory
        cls.tmpdir = tempfile.mkdtemp()
        cls.h5file = os.path.join(cls.tmpdir, f'target_inventory-{_UNIQPIX_SPECPROD}.h5')
        create_inventory(cls.h5file, specprod=_UNIQPIX_SPECPROD, nrows=500)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)

    def test_create_inventory_uniqpix_groups(self):
        self.assertTrue(os.path.exists(self.h5file))
        with h5py.File(self.h5file) as hx:
            self.assertIn('uniqpix_targets', hx)
            self.assertIn('target_uniqpix', hx)
            self.assertIn('target_tiles', hx)
            self.assertIn('hpix2upix', hx)

    def test_uniqpix_target_tiles(self):
        with h5py.File(self.h5file) as hx:
            first_group = next(iter(hx['target_tiles'].keys()))
            tid = int(hx[f'target_tiles/{first_group}']['TARGETID'][0])
        result = target_tiles(targetids=[tid], filename=self.h5file)
        self.assertEqual(len(result), 1)
        self.assertEqual(result['TARGETID'][0], tid)
        for col in ('TILEID', 'LASTNIGHT', 'FIBER'):
            self.assertIn(col, result.colnames)

    def test_radec2targetids_uniqpix(self):
        with h5py.File(self.h5file) as hx:
            first_upix = next(iter(hx['uniqpix_targets'].keys()))
            row = hx[f'uniqpix_targets/{first_upix}'][0]
            tid = int(row['TARGETID'])
            ra  = float(row['TARGET_RA'])
            dec = float(row['TARGET_DEC'])
        targetids = radec2targetids((ra, dec, 120.0), filename=self.h5file)
        self.assertGreater(len(targetids), 0,
            'radec2targetids returned no targets for uniqpix-based inventory')
        self.assertIn(tid, targetids)

    def test_target_healpix_uniqpix(self):
        with h5py.File(self.h5file) as hx:
            first_group = next(iter(hx['target_uniqpix'].keys()))
            tid = int(hx[f'target_uniqpix/{first_group}']['TARGETID'][0])
        result = target_healpix(targetids=[tid], filename=self.h5file)
        self.assertEqual(len(result), 1)
        self.assertEqual(result['TARGETID'][0], tid)
        for col in ('SURVEY', 'PROGRAM', 'UNIQPIX'):
            self.assertIn(col, result.colnames)
        self.assertNotIn('HEALPIX', result.colnames)

    def test_target_healpix_by_radec_uniqpix(self):
        with h5py.File(self.h5file) as hx:
            first_upix = next(iter(hx['uniqpix_targets'].keys()))
            row = hx[f'uniqpix_targets/{first_upix}'][0]
            ra  = float(row['TARGET_RA'])
            dec = float(row['TARGET_DEC'])
        result = target_healpix(radec=(ra, dec, 120.0), filename=self.h5file)
        self.assertGreater(len(result), 0)
        self.assertIn('UNIQPIX', result.colnames)
        self.assertNotIn('HEALPIX', result.colnames)


if __name__ == '__main__':
    unittest.main()
