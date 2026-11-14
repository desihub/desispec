"""
tests desispec.io.write_spectra output compression with pigz, as used by
desispec.scripts.group_spectra
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
from astropy.io.fits.hdu.base import _ValidHDU

from desispec.spectra import Spectra
from desispec.io import empty_fibermap, read_spectra, write_spectra
import desispec.io.util
from desispec.parallel import get_cpu_affinity


@unittest.skipIf(shutil.which('pigz') is None, 'pigz not installed')
class TestWriteSpectraPigz(unittest.TestCase):

    def setUp(self):
        self.testdir = tempfile.mkdtemp()
        self.original_cpus = get_cpu_affinity()
        self.cpus = sorted(self.original_cpus)

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

    def _two_cpus(self):
        """Return two distinct CPU ids this process can run on, or skip the test"""
        if len(self.cpus) < 2:
            self.skipTest('pigz with cpus requires at least 2 CPUs')
        return self.cpus[0:2]

    def test_pigz_output(self):
        """pigz output is a valid gzip file with the same content as write_spectra"""
        #- The byte-identical comparison below requires both writes to produce
        #- identical headers, but astropy puts the current time (to the second)
        #- in the CHECKSUM/DATASUM comments ("HDU checksum updated <timestamp>"),
        #- and CHECKSUM covers the header including that comment.  If the two
        #- writes straddle a second boundary, every CHECKSUM differs and the test
        #- fails intermittently.  Freeze the timestamp for both writes.
        #- This patches the private astropy method _ValidHDU._get_timestamp
        #- (astropy 8.x); if a future astropy renames it, mock.patch.object will
        #- raise AttributeError.  In that case, an alternative is to compare the
        #- decompressed file to the plain one with
        #- fits.FITSDiff(..., ignore_keywords=['CHECKSUM'], ignore_comments=['DATASUM'])
        #- instead of byte-for-byte, at the cost of a weaker check.
        plain = os.path.join(self.testdir, 'plain', 'spectra.fits')
        outfile = os.path.join(self.testdir, 'out', 'spectra.fits.gz')
        with mock.patch.object(_ValidHDU, '_get_timestamp', return_value='2026-01-01T00:00:00'):
            write_spectra(plain, self.spectra)

            with mock.patch.object(desispec.io.util.subprocess, 'Popen',
                                   wraps=subprocess.Popen) as popen:
                write_spectra(outfile, self.spectra, cpus=self._two_cpus())
                popen.assert_called_once()
        self._assert_only_output(outfile)
        self._assert_same_spectra(outfile)

        #- first two bytes of a gzip file
        with open(outfile, 'rb') as fx:
            self.assertEqual(fx.read(2), b'\x1f\x8b')

        #- decompressed bytes are identical to the uncompressed write
        proc = subprocess.run(['gzip', '-dc', outfile], capture_output=True, check=True)
        with open(plain, 'rb') as fx:
            self.assertEqual(proc.stdout, fx.read())

    def test_ncpu(self):
        """ncpu>1 uses pigz without cpus; ncpu=1 doesn't"""
        outfile = os.path.join(self.testdir, 'spectra.fits.gz')
        with mock.patch.object(desispec.io.util.subprocess, 'Popen',
                               wraps=subprocess.Popen) as popen:
            write_spectra(outfile, self.spectra, ncpu=2)
            popen.assert_called_once()
        self._assert_same_spectra(outfile)

        with mock.patch.object(desispec.io.util.subprocess, 'Popen') as popen:
            write_spectra(outfile, self.spectra)
            write_spectra(outfile, self.spectra, ncpu=1)
            write_spectra(outfile, self.spectra, cpus=[])
            popen.assert_not_called()
        self._assert_only_output(outfile)
        self._assert_same_spectra(outfile)

    def test_affinity_restored(self):
        """CPU affinity of the calling process is restored"""
        outfile = os.path.join(self.testdir, 'spectra.fits.gz')
        write_spectra(outfile, self.spectra, cpus=self._two_cpus())
        self.assertEqual(get_cpu_affinity(), self.original_cpus)

    def test_not_gz(self):
        """Files not ending in .gz are written directly"""
        outfile = os.path.join(self.testdir, 'spectra.fits')
        with mock.patch.object(desispec.io.util.subprocess, 'Popen') as popen:
            write_spectra(outfile, self.spectra, cpus=self._two_cpus())
            popen.assert_not_called()
        self._assert_only_output(outfile)
        self._assert_same_spectra(outfile)

    def test_no_pigz(self):
        """Falls back to single-threaded gzip if pigz isn't installed"""
        outfile = os.path.join(self.testdir, 'spectra.fits.gz')
        with mock.patch.object(desispec.io.util.shutil, 'which', return_value=None):
            write_spectra(outfile, self.spectra, cpus=self._two_cpus())
        self._assert_only_output(outfile)
        self._assert_same_spectra(outfile)

    def test_pigz_failure(self):
        """Falls back to single-threaded gzip and cleans up if pigz fails"""
        outfile = os.path.join(self.testdir, 'spectra.fits.gz')
        real_popen = subprocess.Popen
        def failing_pigz(cmd, **kwargs):
            return real_popen(['sh', '-c', 'cat > /dev/null; exit 1'], **kwargs)
        with mock.patch.object(desispec.io.util.subprocess, 'Popen',
                               side_effect=failing_pigz):
            write_spectra(outfile, self.spectra, cpus=self._two_cpus())
        self._assert_only_output(outfile)
        self._assert_same_spectra(outfile)
        self.assertEqual(get_cpu_affinity(), self.original_cpus)


if __name__ == '__main__':
    unittest.main()
