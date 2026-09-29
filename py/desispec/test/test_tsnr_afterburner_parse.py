"""
Tests for the parse() function in bin/desi_tsnr_afterburner.

Phase 1, Step 1.3 of the tsnr_afterburner refactoring plan.
"""

import os
import unittest

import desispec.scripts.tsnr_afterburner as _mod


class TestParse(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mod = _mod

    def _parse(self, args):
        return self.mod.parse(args)

    # ------------------------------------------------------------------
    # Required / basic arguments
    # ------------------------------------------------------------------

    def test_outfile(self):
        args = self._parse(['-o', 'out.fits'])
        self.assertEqual(args.outfile, 'out.fits')

    def test_outfile_long(self):
        args = self._parse(['--outfile', 'out.fits'])
        self.assertEqual(args.outfile, 'out.fits')

    def test_outfile_default_none(self):
        args = self._parse([])
        self.assertIsNone(args.outfile)

    # ------------------------------------------------------------------
    # Flags with defaults
    # ------------------------------------------------------------------

    def test_update_default_false(self):
        args = self._parse([])
        self.assertFalse(args.update)

    def test_update_flag(self):
        args = self._parse(['--update'])
        self.assertTrue(args.update)

    def test_recompute_default_false(self):
        args = self._parse([])
        self.assertFalse(args.recompute)

    def test_recompute_flag(self):
        args = self._parse(['--recompute'])
        self.assertTrue(args.recompute)

    def test_alpha_only_default_false(self):
        args = self._parse([])
        self.assertFalse(args.alpha_only)

    def test_alpha_only_flag(self):
        args = self._parse(['--alpha_only'])
        self.assertTrue(args.alpha_only)

    def test_mpi_default_false(self):
        args = self._parse([])
        self.assertFalse(args.mpi)

    def test_add_badexp_default_false(self):
        args = self._parse([])
        self.assertFalse(args.add_badexp)

    def test_add_badexp_flag(self):
        args = self._parse(['--add-badexp'])
        self.assertTrue(args.add_badexp)

    def test_compute_skymags_default_false(self):
        args = self._parse([])
        self.assertFalse(args.compute_skymags)

    def test_compute_skymags_flag(self):
        args = self._parse(['--compute-skymags'])
        self.assertTrue(args.compute_skymags)

    # ------------------------------------------------------------------
    # Integer / string arguments with defaults
    # ------------------------------------------------------------------

    def test_nproc_default(self):
        args = self._parse([])
        self.assertEqual(args.nproc, 1)

    def test_nproc(self):
        args = self._parse(['--nproc', '32'])
        self.assertEqual(args.nproc, 32)

    def test_prod_default_none(self):
        args = self._parse([])
        self.assertIsNone(args.prod)

    def test_prod(self):
        args = self._parse(['--prod', '/path/to/prod'])
        self.assertEqual(args.prod, '/path/to/prod')

    def test_cameras_default_none(self):
        args = self._parse([])
        self.assertIsNone(args.cameras)

    def test_cameras(self):
        args = self._parse(['-c', 'b0,r0,z0'])
        self.assertEqual(args.cameras, 'b0,r0,z0')

    def test_expids_default_none(self):
        args = self._parse([])
        self.assertIsNone(args.expids)

    def test_expids(self):
        args = self._parse(['-e', '88000,88001'])
        self.assertEqual(args.expids, '88000,88001')

    def test_nights_default_none(self):
        args = self._parse([])
        self.assertIsNone(args.nights)

    def test_nights(self):
        args = self._parse(['-n', '20210401,20210402'])
        self.assertEqual(args.nights, '20210401,20210402')

    def test_details_dir_default_none(self):
        args = self._parse([])
        self.assertIsNone(args.details_dir)

    def test_details_dir(self):
        args = self._parse(['--details-dir', '/some/dir'])
        self.assertEqual(args.details_dir, '/some/dir')

    def test_tile_completeness_default_none(self):
        args = self._parse([])
        self.assertIsNone(args.tile_completeness)

    def test_skymags_default_none(self):
        args = self._parse([])
        self.assertIsNone(args.skymags)

    def test_gfa_proc_dir_default_none(self):
        args = self._parse([])
        self.assertIsNone(args.gfa_proc_dir)

    # ------------------------------------------------------------------
    # nargs="*" argument (aux)
    # ------------------------------------------------------------------

    def test_aux_default_none(self):
        args = self._parse([])
        self.assertIsNone(args.aux)

    def test_aux_single_value(self):
        args = self._parse(['--aux', '/path/to/sv1-tiles.fits'])
        self.assertEqual(args.aux, ['/path/to/sv1-tiles.fits'])

    def test_aux_multiple_values(self):
        args = self._parse(['--aux', '/path/to/sv1.fits', '/path/to/sv2.fits'])
        self.assertEqual(args.aux, ['/path/to/sv1.fits', '/path/to/sv2.fits'])

    def test_aux_zero_values(self):
        # --aux with no arguments returns an empty list
        args = self._parse(['--aux'])
        self.assertEqual(args.aux, [])

    # ------------------------------------------------------------------
    # Combined arguments
    # ------------------------------------------------------------------

    def test_full_typical_invocation(self):
        args = self._parse([
            '-o', '/path/to/out.fits',
            '--prod', 'daily',
            '--nights', '20210401',
            '--nproc', '16',
            '--recompute',
            '--update',
            '--add-badexp',
        ])
        self.assertEqual(args.outfile, '/path/to/out.fits')
        self.assertEqual(args.prod, 'daily')
        self.assertEqual(args.nights, '20210401')
        self.assertEqual(args.nproc, 16)
        self.assertTrue(args.recompute)
        self.assertTrue(args.update)
        self.assertTrue(args.add_badexp)


if __name__ == '__main__':
    unittest.main()
