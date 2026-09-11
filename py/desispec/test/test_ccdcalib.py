import unittest
from unittest.mock import patch, call
import numpy as np
import os


class TestCcdCalib(unittest.TestCase):

    def test_select_zero_expids(self):
        """Test select_zero_expids"""
        original_log_level = os.getenv('DESI_LOGLEVEL')
        os.environ['DESI_LOGLEVEL'] = 'CRITICAL'

        from ..ccdcalib import select_zero_expids
        night = 20000101
        cam = 'b0'

        ## Test standard case of 25 calibs with first two removed
        calib_exps = np.arange(3,26)
        noncalib_exps = np.arange(28, 36)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertListEqual(calib_exps.tolist(),expids.tolist())

        ## Test case of 25 calibs
        calib_exps = np.arange(1,26)
        noncalib_exps = np.arange(30, 40)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertListEqual(calib_exps.tolist(),expids.tolist())

        ## Test case of 28 calibs
        calib_exps = np.arange(1,29)
        noncalib_exps = np.arange(30, 40)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertNotEqual(len(calib_exps),len(expids))
        self.assertEqual(len(expids), 25)

        ## Test case of 15 calibs and plenty of noncals
        calib_exps = np.arange(1,16)
        noncalib_exps = np.arange(30, 40)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertEqual(len(expids),25-2)
        self.assertListEqual(calib_exps.tolist(), expids[:15].tolist())

        ## Test case of 15 calibs and fewer than enough noncals
        calib_exps = np.arange(1,16)
        noncalib_exps = np.arange(30, 36)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertEqual(len(expids),21)
        self.assertListEqual(calib_exps.tolist(), expids[:15].tolist())

        ## Test case of 12 calibs and just enough noncalibs
        calib_exps = np.arange(1,13)
        noncalib_exps = np.arange(30, 52)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertEqual(len(expids),25-2)
        self.assertListEqual(calib_exps.tolist(), expids[:12].tolist())

        ## Test case of 12 calibs and just enough noncalibs
        calib_exps = np.arange(1,13)
        noncalib_exps = np.arange(30, 42)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertEqual(len(expids),25-2)
        self.assertListEqual(calib_exps.tolist(), expids[:12].tolist())

        ## Test case of 12 calibs and less than max noncalibs
        calib_exps = np.arange(1,13)
        noncalib_exps = np.arange(30, 40)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertEqual(len(expids),22)
        self.assertListEqual(calib_exps.tolist(), expids[:12].tolist())

        ## Test case of 15 calibs and no noncalibs
        calib_exps = np.arange(1,16)
        noncalib_exps = np.array([])
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertEqual(len(expids), 15)
        self.assertListEqual(calib_exps.tolist(), expids.tolist())

        ## Test case of 0 calibs and 15 noncalibs
        calib_exps = np.array([])
        noncalib_exps = np.arange(1,16)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertEqual(len(expids), 15)
        self.assertListEqual(noncalib_exps.tolist(), expids.tolist())

        ## Test case of less than minzeros total
        calib_exps = np.arange(1,9)
        noncalib_exps = np.arange(30, 32)
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertIsNone(expids)

        ## Test case of 12 calibs and no noncalibs
        calib_exps = np.arange(1,13)
        noncalib_exps = np.array([])
        expids = select_zero_expids(calib_exps, noncalib_exps, night, cam,
                            nzeros=25, minzeros=15, nskip=2, anyzeros=False)
        self.assertIsNone(expids)

        if original_log_level is None:
            del os.environ['DESI_LOGLEVEL']
        else:
            os.environ['DESI_LOGLEVEL'] = original_log_level


class TestNightlyBiasRetryLoop(unittest.TestCase):
    """Test the nskip retry loop behavior in compute_nightly_bias"""

    def setUp(self):
        self.original_log_level = os.getenv('DESI_LOGLEVEL')
        os.environ['DESI_LOGLEVEL'] = 'CRITICAL'
        self.cameras = ['b0']
        self.night = 20000101
        self.enough_expids = np.arange(1, 16)
        self.calib_dict = {'b0': np.arange(1, 16)}
        self.noncalib_dict = {'b0': np.array([])}

    def tearDown(self):
        if self.original_log_level is None:
            os.environ.pop('DESI_LOGLEVEL', None)
        else:
            os.environ['DESI_LOGLEVEL'] = self.original_log_level

    def _findfile_side_effect(self, *args, **kwargs):
        filetype = args[0] if args else ''
        if filetype == 'biasnight':
            camera = kwargs.get('camera', 'b0')
            night = kwargs.get('night', self.night)
            return f'/tmp/biasnight-{night}-{camera}.fits'
        return '/tmp/raw-test.fits'

    def test_raises_when_no_nskip_works(self):
        """RuntimeError is raised when no nskip value provides enough ZEROs"""
        from ..ccdcalib import compute_nightly_bias

        with patch('desispec.ccdcalib._find_zeros',
                   return_value=(self.calib_dict, self.noncalib_dict)), \
             patch('desispec.ccdcalib.select_zero_expids', return_value=None):
            with self.assertRaises(RuntimeError):
                compute_nightly_bias(self.night, self.cameras, nskip=2)

    def test_retries_with_decreasing_nskip(self):
        """Loop decrements nskip and stops when all cameras have enough ZEROs"""
        from ..ccdcalib import compute_nightly_bias

        def select_side_effect(calib_exps, noncalib_exps, night, cam,
                               nzeros, minzeros, nskip, anyzeros):
            return None if nskip > 0 else self.enough_expids

        with patch('desispec.ccdcalib._find_zeros',
                   return_value=(self.calib_dict, self.noncalib_dict)) as mock_find, \
             patch('desispec.ccdcalib.select_zero_expids',
                   side_effect=select_side_effect), \
             patch('desispec.io.findfile',
                   side_effect=self._findfile_side_effect), \
             patch('os.makedirs'), \
             patch('os.path.exists',
                   side_effect=lambda p: 'biasnighttest-' not in p):
            compute_nightly_bias(self.night, self.cameras, nskip=2)

        expected_calls = [
            call(self.night, cameras=self.cameras, nzeros=25, nskip=2),
            call(self.night, cameras=self.cameras, nzeros=25, nskip=1),
            call(self.night, cameras=self.cameras, nzeros=25, nskip=0),
        ]
        mock_find.assert_has_calls(expected_calls)
        self.assertEqual(mock_find.call_count, 3)

    def test_stops_when_all_cameras_succeed(self):
        """Loop stops at nskip=2 when all cameras succeed on first attempt"""
        from ..ccdcalib import compute_nightly_bias

        with patch('desispec.ccdcalib._find_zeros',
                   return_value=(self.calib_dict, self.noncalib_dict)) as mock_find, \
             patch('desispec.ccdcalib.select_zero_expids',
                   return_value=self.enough_expids), \
             patch('desispec.io.findfile',
                   side_effect=self._findfile_side_effect), \
             patch('os.makedirs'), \
             patch('os.path.exists',
                   side_effect=lambda p: 'biasnighttest-' not in p):
            compute_nightly_bias(self.night, self.cameras, nskip=2)

        mock_find.assert_called_once_with(
            self.night, cameras=self.cameras, nzeros=25, nskip=2)


class TestDarkPreprocBias(unittest.TestCase):
    """Test identifying which bias a preprocessed dark was created with"""

    def setUp(self):
        self.original_log_level = os.getenv('DESI_LOGLEVEL')
        os.environ['DESI_LOGLEVEL'] = 'CRITICAL'
        self.night = 20000101
        self.camera = 'b0'
        self.biasnight = f'/tmp/calibnight/{self.night}/biasnight-{self.camera}-{self.night}.fits.gz'

    def tearDown(self):
        if self.original_log_level is None:
            os.environ.pop('DESI_LOGLEVEL', None)
        else:
            os.environ['DESI_LOGLEVEL'] = self.original_log_level

    def _header(self, biasused):
        from astropy.io.fits import Header
        from desiutil.depend import setdep
        header = Header()
        if biasused is not None:
            setdep(header, 'CCD_CALIB_BIAS', biasused)
        return header

    def _is_nightly(self, biasused, night=None, camera=None):
        from ..ccdcalib import dark_preproc_bias_is_nightly
        if night is None:
            night = self.night
        if camera is None:
            camera = self.camera
        with patch('desispec.ccdcalib.findfile', return_value=self.biasnight):
            return dark_preproc_bias_is_nightly(self._header(biasused), night, camera)

    def test_nightly_bias(self):
        """The nightly bias of this night and camera is accepted"""
        biasused = f'SPECPROD/calibnight/{self.night}/biasnight-{self.camera}-{self.night}.fits.gz'
        is_nightly, found = self._is_nightly(biasused)
        self.assertTrue(is_nightly)
        self.assertEqual(found, biasused)

    def test_uncompressed_nightly_bias(self):
        """An uncompressed nightly bias is still recognized"""
        biasused = f'SPECPROD/calibnight/{self.night}/biasnight-{self.camera}-{self.night}.fits'
        is_nightly, found = self._is_nightly(biasused)
        self.assertTrue(is_nightly)

    def test_default_bias(self):
        """The default bias in DESI_SPECTRO_CALIB is rejected"""
        biasused = 'SPCALIB/ccd/bias-sm4-b-20191021.fits.gz'
        is_nightly, found = self._is_nightly(biasused)
        self.assertFalse(is_nightly)
        self.assertEqual(found, biasused)

    def test_other_night_and_camera(self):
        """A nightly bias of another night or camera is rejected"""
        biasused = f'SPECPROD/calibnight/{self.night}/biasnight-{self.camera}-{self.night}.fits.gz'
        with patch('desispec.ccdcalib.findfile',
                   return_value='/tmp/calibnight/20000102/biasnight-b0-20000102.fits.gz'):
            from ..ccdcalib import dark_preproc_bias_is_nightly
            is_nightly, found = dark_preproc_bias_is_nightly(
                    self._header(biasused), 20000102, 'b0')
        self.assertFalse(is_nightly)

    def test_explicitly_requested_bias(self):
        """An explicitly requested bias is matched by filename"""
        from ..ccdcalib import dark_preproc_bias_matches
        header = self._header('SPCALIB/ccd/bias-sm4-b-20191021.fits.gz')
        is_match, found = dark_preproc_bias_matches(header, '/tmp/bias-sm4-b-20191021.fits.gz')
        self.assertTrue(is_match)
        is_match, found = dark_preproc_bias_matches(header, '/tmp/bias-sm4-b-20200401.fits.gz')
        self.assertFalse(is_match)

    def test_expected_bias_per_night(self):
        """Each night of a multi-night dark expects its own nightly bias

        desi_compute_dark_night spans ~45 nights around a reference night, so
        expecting one night's bias for all of them would reject every exposure
        except those of that night (desispec issue #2741).
        """
        from ..ccdcalib import expected_dark_preproc_bias

        def findfile_side_effect(filetype, night=None, camera=None, **kwargs):
            self.assertEqual(filetype, 'biasnight')
            return f'/tmp/calibnight/{night}/biasnight-{camera}-{night}.fits.gz'

        refnight = 20000131
        othernight = 20000101
        with patch('desispec.ccdcalib.findfile', side_effect=findfile_side_effect):
            #- bias=True lets each exposure resolve its own night
            refbias = expected_dark_preproc_bias(True, refnight, self.camera)
            otherbias = expected_dark_preproc_bias(True, othernight, self.camera)
            self.assertIn(f'biasnight-{self.camera}-{refnight}', refbias)
            self.assertIn(f'biasnight-{self.camera}-{othernight}', otherbias)
            self.assertNotEqual(refbias, otherbias)

            #- a camera expects its own bias, not another camera's
            othercam = expected_dark_preproc_bias(True, refnight, 'z9')
            self.assertIn(f'biasnight-z9-{refnight}', othercam)

            #- an explicitly requested bias is used as given for every night
            custom = '/tmp/mybias.fits'
            self.assertEqual(expected_dark_preproc_bias(custom, refnight, self.camera), custom)
            self.assertEqual(expected_dark_preproc_bias(custom, othernight, self.camera), custom)

    def test_no_bias_recorded(self):
        """A header without CCD_CALIB_BIAS is rejected"""
        is_nightly, found = self._is_nightly(None)
        self.assertFalse(is_nightly)
        self.assertIsNone(found)

    def test_unreadable_file(self):
        """An unreadable file is rejected instead of raising"""
        from ..ccdcalib import dark_preproc_bias_is_nightly
        is_nightly, found = dark_preproc_bias_is_nightly(
                '/tmp/does-not-exist-dark_preproc-b0-00000001.fits', self.night, self.camera)
        self.assertFalse(is_nightly)
        self.assertIsNone(found)
