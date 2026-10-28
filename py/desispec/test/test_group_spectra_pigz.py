"""
tests desispec.scripts.group_spectra output compression with pigz
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from desispec.spectra import Spectra
from desispec.io import empty_fibermap, read_spectra, write_spectra
from desispec.scripts import group_spectra


@unittest.skipIf(shutil.which('pigz') is None, 'pigz not installed')
class TestWriteSpectraPigz(unittest.TestCase):

    def setUp(self):
        self.testdir = tempfile.mkdtemp()
        self.cpus = sorted(os.sched_getaffinity(0))
        self.original_cpus = os.sched_getaffinity(0)

        bands = ['b', 'r', 'z']
        wave = {'b': np.arange(3600, 5800.8, 0.8),
                'r': np.arange(5760, 7620.8, 0.8),
                'z': np.arange(7520, 9824.8, 0.8)}
        rng = np.random.default_rng(12)
        nspec = 5
        flux = {b: rng.normal(size=(nspec, len(wave[b]))) for b in bands}
        ivar = {b: rng.uniform(1, 2, size=(nspec, len(wave[b]))) for b in bands}
        rdat = {b: np.ones((nspec, 1, len(wave[b]))) for b in bands}
        fmap = empty_fibermap(nspec)
        fmap['TARGETID'] = 100 + np.arange(nspec)
        fmap['TILEID'] = 1000
        fmap['NIGHT'] = 20200101
        fmap['EXPID'] = 5000
        fmap['TARGET_RA'] += 10
        fmap['FIBER_RA'] += 10
        self.spectra = Spectra(bands=bands, wave=wave, flux=flux, ivar=ivar,
                               mask=None, resolution_data=rdat, fibermap=fmap)

    def tearDown(self):
        os.sched_setaffinity(0, self.original_cpus)
        shutil.rmtree(self.testdir, ignore_errors=True)

    def _assert_same_spectra(self, filename):
        """Check that filename has the same data as self.spectra"""
        spec = read_spectra(filename)
        for band in self.spectra.bands:
            np.testing.assert_array_equal(spec.wave[band], self.spectra.wave[band])
            np.testing.assert_array_almost_equal(spec.flux[band], self.spectra.flux[band])
            np.testing.assert_array_almost_equal(spec.ivar[band], self.spectra.ivar[band])
        np.testing.assert_array_equal(spec.fibermap['TARGETID'],
                                      self.spectra.fibermap['TARGETID'])

    def _assert_only_output(self, filename):
        """Check that no uncompressed or temporary files were left behind"""
        self.assertEqual(os.listdir(os.path.dirname(filename)),
                         [os.path.basename(filename)])

    def test_pigz_output(self):
        """pigz output is a valid gzip file with the same content as write_spectra"""
        reference = os.path.join(self.testdir, 'ref', 'spectra.fits.gz')
        write_spectra(reference, self.spectra)

        outfile = os.path.join(self.testdir, 'out', 'spectra.fits.gz')
        group_spectra._write_spectra_pigz(outfile, self.spectra, self.cpus, min_cpus=1)
        self._assert_only_output(outfile)
        self._assert_same_spectra(outfile)

        #- first two bytes of a gzip file
        with open(outfile, 'rb') as fx:
            self.assertEqual(fx.read(2), b'\x1f\x8b')

        #- decompressed bytes are the same size as the uncompressed write
        plain = os.path.join(self.testdir, 'plain', 'spectra.fits')
        write_spectra(plain, self.spectra)
        proc = subprocess.run(['gzip', '-dc', outfile], capture_output=True, check=True)
        self.assertEqual(len(proc.stdout), os.path.getsize(plain))

        #- same content as the standard writer
        a = read_spectra(reference)
        b = read_spectra(outfile)
        for band in a.bands:
            np.testing.assert_array_equal(a.flux[band], b.flux[band])
            np.testing.assert_array_equal(a.ivar[band], b.ivar[band])

    def test_affinity_restored(self):
        """CPU affinity of the calling process is restored"""
        outfile = os.path.join(self.testdir, 'spectra.fits.gz')
        group_spectra._write_spectra_pigz(outfile, self.spectra, self.cpus, min_cpus=1)
        self.assertEqual(os.sched_getaffinity(0), self.original_cpus)

    def test_not_gz(self):
        """Files not ending in .gz are written directly"""
        outfile = os.path.join(self.testdir, 'spectra.fits')
        group_spectra._write_spectra_pigz(outfile, self.spectra, self.cpus, min_cpus=1)
        self._assert_only_output(outfile)
        self._assert_same_spectra(outfile)

    def test_few_cpus(self):
        """Falls back to write_spectra if fewer than min_cpus CPUs are available"""
        outfile = os.path.join(self.testdir, 'spectra.fits.gz')
        with mock.patch.object(group_spectra.subprocess, 'run') as run:
            group_spectra._write_spectra_pigz(outfile, self.spectra, self.cpus[:1], min_cpus=2)
            run.assert_not_called()
        self._assert_only_output(outfile)
        self._assert_same_spectra(outfile)

    def test_no_pigz(self):
        """Falls back to write_spectra if pigz isn't installed"""
        outfile = os.path.join(self.testdir, 'spectra.fits.gz')
        with mock.patch.object(group_spectra.shutil, 'which', return_value=None):
            group_spectra._write_spectra_pigz(outfile, self.spectra, self.cpus, min_cpus=1)
        self._assert_only_output(outfile)
        self._assert_same_spectra(outfile)

    def test_pigz_failure(self):
        """Falls back to write_spectra and cleans up if pigz fails"""
        outfile = os.path.join(self.testdir, 'spectra.fits.gz')
        err = subprocess.CalledProcessError(1, 'pigz')
        with mock.patch.object(group_spectra.subprocess, 'run', side_effect=err):
            group_spectra._write_spectra_pigz(outfile, self.spectra, self.cpus, min_cpus=1)
        self._assert_only_output(outfile)
        self._assert_same_spectra(outfile)
        self.assertEqual(os.sched_getaffinity(0), self.original_cpus)


if __name__ == '__main__':
    unittest.main()
