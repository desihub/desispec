"""
test desispec.scripts.preproc_darks_mpi
"""

import os
import unittest
from unittest.mock import patch

import numpy as np
from astropy.table import Table


def _findfile(filetype, night=None, expid=None, camera=None, **kwargs):
    """Stand-in for desispec.io.findfile with predictable paths"""
    if filetype == 'biasnight':
        return f'/tmp/calibnight/{night}/biasnight-{camera}-{night}.fits.gz'
    elif filetype == 'preproc_for_dark':
        return f'/tmp/dark_preproc/{night}/{expid:08d}/dark_preproc-{camera}-{expid:08d}.fits'
    elif filetype == 'raw':
        return f'/tmp/{night}/{expid:08d}/desi-{expid:08d}.fits.fz'
    elif filetype == 'exposure_table':
        return f'/tmp/exposure_tables/exposure_table_{night}.csv'
    else:
        raise ValueError(f'Unexpected {filetype=}')


class TestCheckMatchingBiasnights(unittest.TestCase):
    """Test that missing or mismatched nightly biases are reported"""

    def setUp(self):
        self.original_log_level = os.getenv('DESI_LOGLEVEL')
        os.environ['DESI_LOGLEVEL'] = 'CRITICAL'
        self.night = 20000101
        self.expids = [10, 11]
        self.nights = [self.night, self.night]
        self.camlists = [['b0', 'b1'], ['b0', 'b1']]
        self.rawfiles = [_findfile('raw', night=self.night, expid=e) for e in self.expids]
        #- by default the raw headers agree with the exposure table
        header_patcher = patch('desispec.scripts.preproc_darks_mpi.read_raw_primary_header',
                               return_value={'NIGHT': self.night})
        header_patcher.start()
        self.addCleanup(header_patcher.stop)

    def tearDown(self):
        if self.original_log_level is None:
            os.environ.pop('DESI_LOGLEVEL', None)
        else:
            os.environ['DESI_LOGLEVEL'] = self.original_log_level

    def test_no_errors_when_biasnights_exist(self):
        """No errors when every camera has a biasnight and no preproc exists yet"""
        from ..scripts.preproc_darks_mpi import check_matching_biasnights

        with patch('desispec.scripts.preproc_darks_mpi.findfile', side_effect=_findfile), \
             patch('os.path.exists', side_effect=lambda p: 'biasnight-' in p):
            errors = check_matching_biasnights(self.expids, self.nights, self.camlists, self.rawfiles)

        self.assertEqual(errors, [])

    def test_missing_biasnight(self):
        """A camera without a biasnight is reported for every exposure"""
        from ..scripts.preproc_darks_mpi import check_matching_biasnights

        def exists(path):
            return 'biasnight-' in path and 'biasnight-b0-' not in path

        with patch('desispec.scripts.preproc_darks_mpi.findfile', side_effect=_findfile), \
             patch('os.path.exists', side_effect=exists):
            errors = check_matching_biasnights(self.expids, self.nights, self.camlists, self.rawfiles)

        self.assertEqual(len(errors), len(self.expids))
        for msg in errors:
            self.assertIn(f'biasnight-b0-{self.night}', msg)
            self.assertIn('Missing', msg)
            self.assertNotIn('rejected', msg)

    def test_rejected_biasnight(self):
        """A biasnighttest file is called out as a rejected nightly bias"""
        from ..scripts.preproc_darks_mpi import check_matching_biasnights

        def exists(path):
            if 'biasnighttest-b0-' in path:
                return True
            return 'biasnight-' in path and 'biasnight-b0-' not in path

        with patch('desispec.scripts.preproc_darks_mpi.findfile', side_effect=_findfile), \
             patch('os.path.exists', side_effect=exists):
            errors = check_matching_biasnights(self.expids, self.nights, self.camlists, self.rawfiles)

        self.assertEqual(len(errors), len(self.expids))
        for msg in errors:
            self.assertIn(f'biasnighttest-b0-{self.night}', msg)
            self.assertIn('rejected', msg)

    def test_existing_preproc_with_default_bias(self):
        """Pre-existing preprocs made with another bias are reported, not overwritten"""
        from ..scripts.preproc_darks_mpi import check_matching_biasnights

        with patch('desispec.scripts.preproc_darks_mpi.findfile', side_effect=_findfile), \
             patch('os.path.exists', return_value=True), \
             patch('desispec.scripts.preproc_darks_mpi.dark_preproc_bias_is_nightly',
                   return_value=(False, 'SPCALIB/ccd/bias-sm4-b-20191021.fits.gz')):
            errors = check_matching_biasnights(self.expids, self.nights, self.camlists, self.rawfiles)

        self.assertEqual(len(errors), 4)  #- 2 exposures x 2 cameras
        for msg in errors:
            self.assertIn('bias-sm4-b-20191021.fits.gz', msg)
            self.assertIn('purge', msg)

    def test_raw_header_night_mismatch(self):
        """A raw header night that disagrees with the exposure table is rejected

        preproc would look up the bias with the header night while the file is
        written under the exposure table night, so nothing should be written.
        """
        from ..scripts.preproc_darks_mpi import check_matching_biasnights

        with patch('desispec.scripts.preproc_darks_mpi.findfile', side_effect=_findfile), \
             patch('desispec.scripts.preproc_darks_mpi.read_raw_primary_header',
                   return_value={'NIGHT': self.night - 1}), \
             patch('os.path.exists', side_effect=lambda p: 'biasnight-' in p):
            errors = check_matching_biasnights(self.expids, self.nights, self.camlists,
                                               self.rawfiles)

        self.assertEqual(len(errors), len(self.expids))
        for msg in errors:
            self.assertIn(f'NIGHT={self.night - 1}', msg)
            self.assertIn(f'NIGHT={self.night}', msg)

    def test_unreadable_raw_header(self):
        """An unreadable raw header is an error rather than a crash"""
        from ..scripts.preproc_darks_mpi import check_matching_biasnights

        with patch('desispec.scripts.preproc_darks_mpi.findfile', side_effect=_findfile), \
             patch('desispec.scripts.preproc_darks_mpi.read_raw_primary_header',
                   side_effect=OSError('no such file')), \
             patch('os.path.exists', side_effect=lambda p: 'biasnight-' in p):
            errors = check_matching_biasnights(self.expids, self.nights, self.camlists,
                                               self.rawfiles)

        self.assertEqual(len(errors), len(self.expids))
        for msg in errors:
            self.assertIn('Unable to read the night', msg)

    def test_existing_preproc_without_biasnight(self):
        """An existing matching preproc is enough even if the biasnight is gone"""
        from ..scripts.preproc_darks_mpi import check_matching_biasnights

        with patch('desispec.scripts.preproc_darks_mpi.findfile', side_effect=_findfile), \
             patch('os.path.exists', side_effect=lambda p: 'dark_preproc-' in p), \
             patch('desispec.scripts.preproc_darks_mpi.dark_preproc_bias_is_nightly',
                   return_value=(True, 'SPECPROD/calibnight/20000101/biasnight-b0-20000101.fits.gz')):
            errors = check_matching_biasnights(self.expids, self.nights, self.camlists, self.rawfiles)

        self.assertEqual(errors, [])

    def test_existing_preproc_with_nightly_bias(self):
        """Pre-existing preprocs made with the matching nightly bias are fine"""
        from ..scripts.preproc_darks_mpi import check_matching_biasnights

        with patch('desispec.scripts.preproc_darks_mpi.findfile', side_effect=_findfile), \
             patch('os.path.exists', return_value=True), \
             patch('desispec.scripts.preproc_darks_mpi.dark_preproc_bias_is_nightly',
                   return_value=(True, f'SPECPROD/calibnight/{20000101}/biasnight-b0-20000101.fits.gz')):
            errors = check_matching_biasnights(self.expids, self.nights, self.camlists, self.rawfiles)

        self.assertEqual(errors, [])


class TestPreprocDarksMain(unittest.TestCase):
    """Test that main() refuses to preprocess without matching nightly biases"""

    def setUp(self):
        self.original_log_level = os.getenv('DESI_LOGLEVEL')
        os.environ['DESI_LOGLEVEL'] = 'CRITICAL'
        self.night = 20000101
        self.expids = [10, 11]
        self.exptable = Table(dict(
            NIGHT=np.array([self.night, self.night]),
            EXPID=np.array(self.expids),
            CAMWORD=np.array(['a01', 'a01']),
            BADCAMWORD=np.array(['', '']),
            ))

    def tearDown(self):
        if self.original_log_level is None:
            os.environ.pop('DESI_LOGLEVEL', None)
        else:
            os.environ['DESI_LOGLEVEL'] = self.original_log_level

    def _run_main(self, exists, extra_options=None):
        """Run main() with mocked I/O, returning (exitcode, process_raw, write_image)"""
        from ..scripts import preproc_darks_mpi

        options = ['-e', ','.join([str(e) for e in self.expids]),
                   '-n', str(self.night), '-c', 'b01']
        if extra_options is not None:
            options.extend(extra_options)

        with patch('desispec.scripts.preproc_darks_mpi.findfile', side_effect=_findfile), \
             patch('desispec.scripts.preproc_darks_mpi.load_table', return_value=self.exptable), \
             patch('os.path.exists', side_effect=exists), \
             patch('desispec.scripts.preproc_darks_mpi.read_raw_primary_header',
                   return_value={'NIGHT': self.night}), \
             patch('desispec.scripts.preproc_darks_mpi.process_raw') as process_raw, \
             patch('desispec.scripts.preproc_darks_mpi.write_image') as write_image:
            exitcode = preproc_darks_mpi.main(preproc_darks_mpi.parse(options))

        return exitcode, process_raw, write_image

    def test_exits_nonzero_for_one_missing_biasnight(self):
        """One camera without a biasnight stops all of the preprocessing"""
        def exists(path):
            #- raw data and the b1 biasnight exist, the b0 biasnight does not
            return 'desi-' in path or 'biasnight-b1-' in path

        exitcode, process_raw, write_image = self._run_main(exists)

        self.assertEqual(exitcode, 1)
        process_raw.assert_not_called()
        write_image.assert_not_called()

    def test_proceeds_when_all_biasnights_exist(self):
        """The strict check passes when every camera has its own biasnight"""
        def exists(path):
            return 'desi-' in path or 'biasnight-' in path

        exitcode, process_raw, write_image = self._run_main(exists,
                                                            extra_options=['--dry-run'])

        self.assertEqual(exitcode, 0)

    def test_allow_default_bias_skips_the_check(self):
        """--allow-default-bias restores the previous fallback behavior"""
        def exists(path):
            #- neither biasnight exists, and no preproc has been written yet
            return 'desi-' in path

        exitcode, process_raw, write_image = self._run_main(
                exists, extra_options=['--allow-default-bias', '--dry-run'])

        #- --dry-run exits after the check, which shouldn't have failed
        self.assertEqual(exitcode, 0)


if __name__ == '__main__':
    unittest.main()
