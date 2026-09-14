"""
test desispec.scripts.compute_dark_night
"""

import os
import unittest
from unittest.mock import patch

import numpy as np
from astropy.table import Table


class TestComputeDarkNight(unittest.TestCase):
    """Test how desi_compute_dark_night hands the bias to desi_compute_dark"""

    def setUp(self):
        self.original_log_level = os.getenv('DESI_LOGLEVEL')
        os.environ['DESI_LOGLEVEL'] = 'CRITICAL'
        self.refnight = 20000131
        #- darks spanning several nights, as get_stacked_dark_exposure_table returns
        self.nights = [20000101, 20000115, self.refnight, 20000210, 20000215, 20000220]
        self.exptable = Table(dict(
            NIGHT=np.array(self.nights),
            EXPID=np.arange(10, 10+len(self.nights)),
            ))

    def tearDown(self):
        if self.original_log_level is None:
            os.environ.pop('DESI_LOGLEVEL', None)
        else:
            os.environ['DESI_LOGLEVEL'] = self.original_log_level

    def _findfile(self, filetype, night=None, camera=None, **kwargs):
        if filetype == 'biasnight':
            return f'/tmp/calibnight/{night}/biasnight-{camera}-{night}.fits.gz'
        elif filetype == 'darknight':
            return f'/tmp/calibnight/{night}/darknight-{camera}-{night}.fits.gz'
        else:
            raise ValueError(f'Unexpected {filetype=}')

    def test_allow_default_bias_drops_the_required_bias(self):
        """--allow-default-bias also stops requiring this night's bias to exist

        Otherwise runcmd would return before compute_dark ever runs, in exactly
        the missing-bias case the option exists to allow.
        """
        exitcode, calls = self._run_main(extra_options=['--allow-default-bias'])

        self.assertEqual(exitcode, 0)
        for call in calls:
            self.assertEqual(call['inputs'], [])
            self.assertIsNone(call['bias'])

    def _run_main(self, cameras='b0b1', extra_options=None):
        """Run main() with runcmd mocked, returning what it passed to compute_dark"""
        from ..scripts import compute_dark_night

        calls = []

        def fake_runcmd(func, args=None, expandargs=False, inputs=[], outputs=[], **kwargs):
            darkargs, darkexptable = args
            calls.append(dict(bias=darkargs.bias, camera=darkargs.camera,
                              inputs=list(inputs), outputs=list(outputs)))
            return None, True

        options = ['--reference-night', str(self.refnight), '-c', cameras]
        if extra_options is not None:
            options.extend(extra_options)
        with patch('desispec.scripts.compute_dark_night.runcmd', side_effect=fake_runcmd), \
             patch('desispec.scripts.compute_dark_night.compute_dark.get_stacked_dark_exposure_table',
                   return_value=self.exptable), \
             patch('desispec.scripts.compute_dark_night.findfile', side_effect=self._findfile):
            exitcode = compute_dark_night.main(compute_dark_night.parse(options))

        return exitcode, calls

    def test_bias_is_not_pinned_to_the_reference_night(self):
        """compute_dark must resolve a bias per exposure, not get one for all nights

        The darks span ~45 nights, each with its own biasnight, so pinning the
        reference night's bias would mean every other night's preprocessed dark
        looks like it used the wrong bias (desispec issue #2741).
        """
        exitcode, calls = self._run_main()

        self.assertEqual(exitcode, 0)
        self.assertEqual(len(calls), 2)  #- one per camera
        for call in calls:
            self.assertIsNone(call['bias'],
                              'compute_dark_night must leave the bias unset so that '
                              'compute_dark_file uses each exposure\'s own nightly bias')

    def test_reference_night_bias_is_still_required(self):
        """The reference night's bias remains an input that must exist"""
        exitcode, calls = self._run_main()

        for call in calls:
            self.assertEqual(len(call['inputs']), 1)
            self.assertIn(f'biasnight-{call["camera"]}-{self.refnight}', call['inputs'][0])

    def test_explicit_bias_is_passed_through(self):
        """An explicitly requested bias is used and required for every camera"""
        from ..scripts import compute_dark_night

        calls = []

        def fake_runcmd(func, args=None, expandargs=False, inputs=[], outputs=[], **kwargs):
            darkargs, darkexptable = args
            calls.append(dict(bias=darkargs.bias, inputs=list(inputs)))
            return None, True

        options = ['--reference-night', str(self.refnight), '-c', 'b0',
                   '--bias', '/tmp/mybias.fits']
        with patch('desispec.scripts.compute_dark_night.runcmd', side_effect=fake_runcmd), \
             patch('desispec.scripts.compute_dark_night.compute_dark.get_stacked_dark_exposure_table',
                   return_value=self.exptable), \
             patch('desispec.scripts.compute_dark_night.findfile', side_effect=self._findfile):
            compute_dark_night.main(compute_dark_night.parse(options))

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['bias'], '/tmp/mybias.fits')
        self.assertEqual(calls[0]['inputs'], ['/tmp/mybias.fits'])


if __name__ == '__main__':
    unittest.main()
