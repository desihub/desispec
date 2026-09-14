"""
Test desispec.scripts.specex
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from desispec.scripts.specex import merge_psf, mean_psf

#- Fiber status codes used by specex PSF files:
#-   -1 : fiber not part of this bundle (not applicable)
#-    0 : fiber successfully fit
#-   >0 : fiber fit failed with this error code
NOT_APPLICABLE = -1
FIT_OK = 0
FIT_FAILED = 2
#- status written by the specex QA for fibers whose traces cross each other;
#- see the z7 fit failures of night 20250822 (desihub/specex#91)
CROSSED_TRACES = 4


def _write_psf(filename, status, bundle, legcoeff, xtrace, ytrace,
               fit_bundles=(), param_order=('STATUS', 'BUNDLE', 'LEGCOEFF'),
               rchi2=None, header=None):
    """
    Write a minimal specex-like PSF fits file for testing merge_psf/mean_psf.

    Args:
        filename: output path
        status: per-fiber STATUS values, 1D array of length nfibers
        bundle: per-fiber BUNDLE values, 1D array of length nfibers
        legcoeff: per-fiber "trace" legendre coefficients, 2D (nfibers, ncoeff)
        xtrace, ytrace: 2D (nfibers, ncoeff) arrays
        fit_bundles: bundle ids that were actually fit in this file; merge_psf
            requires B{bundle:02d}RCHI2/NDATA/NPAR header keys for these.
            Ignored when rchi2 is given.
        param_order: names and order of the PARAM rows. Real files put STATUS
            near the end, so tests can use this to confirm that results don't
            depend on which row happens to be first.
        rchi2: per-bundle B{bundle:02d}RCHI2 values, length nbundles. Keys are
            written for every bundle including those with rchi2=0, because
            mean_psf counts bundles by scanning B00RCHI2, B01RCHI2, ... and
            stops at the first missing key.
        header: optional dict of extra PSF header keywords
    """
    nfibers = len(status)
    ncoeff = legcoeff.shape[1]

    values = dict(STATUS=status, BUNDLE=bundle, LEGCOEFF=legcoeff)
    param = np.array(list(param_order))
    coeff = np.zeros((len(param), nfibers, ncoeff))
    for i, name in enumerate(param_order):
        if name == 'LEGCOEFF':
            coeff[i] = values[name]
        else:
            coeff[i, :, 0] = values[name]

    col_param = fits.Column(name='PARAM', format='15A', array=param)
    col_coeff = fits.Column(name='COEFF', format='{}D'.format(nfibers * ncoeff),
                             dim='({},{})'.format(ncoeff, nfibers), array=coeff)
    psf_hdu = fits.BinTableHDU.from_columns([col_param, col_coeff], name='PSF')
    psf_hdu.header['FIBERMIN'] = 0
    psf_hdu.header['FIBERMAX'] = nfibers - 1
    if rchi2 is None:
        for b in fit_bundles:
            psf_hdu.header['B{:02d}RCHI2'.format(b)] = 1.0
            psf_hdu.header['B{:02d}NDATA'.format(b)] = 100
            psf_hdu.header['B{:02d}NPAR'.format(b)] = 10
    else:
        for b, value in enumerate(rchi2):
            psf_hdu.header['B{:02d}RCHI2'.format(b)] = float(value)
            psf_hdu.header['B{:02d}NDATA'.format(b)] = 100
            psf_hdu.header['B{:02d}NPAR'.format(b)] = 10
    if header is not None:
        for key, value in header.items():
            psf_hdu.header[key] = value

    xtrace_hdu = fits.ImageHDU(xtrace, name='XTRACE')
    ytrace_hdu = fits.ImageHDU(ytrace, name='YTRACE')

    hdulist = fits.HDUList([fits.PrimaryHDU(), psf_hdu, xtrace_hdu, ytrace_hdu])
    hdulist.writeto(filename, overwrite=True)


def _write_mean_psf_input(filename, status, bundle, rchi2, value, ncoeff=2,
                          param_order=('LEGCOEFF', 'BUNDLE', 'STATUS')):
    """
    Write a merged per-exposure PSF (fit-psf-CAM-EXPID.fits) for mean_psf.

    Args:
        filename: output path
        status: per-fiber STATUS values, 1D array of length nfibers
        bundle: per-fiber BUNDLE values, 1D array of length nfibers
        rchi2: per-bundle rchi2, 0.0 for bundles that are missing or failed
        value: scalar filled into LEGCOEFF/XTRACE/YTRACE so that which input
            was averaged or selected can be read straight off the output
        ncoeff: number of legendre coefficients per fiber
        param_order: see _write_psf; the default puts STATUS last as real
            specex files do

    The header keywords are the ones mean_psf reads directly (PSFVER, CAMERA,
    WAVEMIN, WAVEMAX) plus those compared by specex.compatible(), which must
    match across all inputs or mean_psf calls sys.exit(12).
    """
    nfibers = len(status)
    payload = np.full((nfibers, ncoeff), float(value))
    header = dict(PSFVER='3', CAMERA="'z7      '",
                  WAVEMIN=3526.0, WAVEMAX=6055.0,
                  PSFTYPE='GAUSS-HERMITE', NPIX_X=4096, NPIX_Y=4096,
                  HSIZEX=8, HSIZEY=5, NPARAMS=57, LEGDEG=ncoeff - 1,
                  GHDEGX=6, GHDEGY=6)
    _write_psf(filename, status, bundle, payload, payload, payload,
               param_order=param_order, rchi2=rchi2, header=header)


class TestMergePSF(unittest.TestCase):

    def setUp(self):
        self.testdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.testdir, ignore_errors=True)

    def test_failed_fibers_status_written_before_continue(self):
        """
        A bundle where every fiber fails to fit has no "selected" fibers,
        so merge_psf hits its `continue` for that bundle without copying
        xtrace/ytrace/other parameters. Regardless, the STATUS for those
        failed fibers must still be recorded in the merged output instead
        of being skipped over by that `continue`.
        """
        nfibers = 4
        ncoeff = 2

        #- reference psf covering all fibers; STATUS gets reset to -1
        #- internally by merge_psf for every fiber before merging in inputs
        ref_status = np.zeros(nfibers)
        ref_bundle = np.array([0, 0, 1, 1])
        ref_legcoeff = np.zeros((nfibers, ncoeff))
        ref_xtrace = np.zeros((nfibers, ncoeff))
        ref_ytrace = np.zeros((nfibers, ncoeff))
        reffile = os.path.join(self.testdir, 'psf-ref.fits')
        _write_psf(reffile, ref_status, ref_bundle, ref_legcoeff,
                   ref_xtrace, ref_ytrace)

        #- bundle 0 (fibers 0,1): fit succeeded
        b0_status = np.array([FIT_OK, FIT_OK, NOT_APPLICABLE, NOT_APPLICABLE])
        b0_bundle = np.array([0, 0, -1, -1])
        b0_legcoeff = np.full((nfibers, ncoeff), 1.0)
        b0_xtrace = np.full((nfibers, ncoeff), 11.0)
        b0_ytrace = np.full((nfibers, ncoeff), 12.0)
        b0file = os.path.join(self.testdir, 'psf-bundle0.fits')
        _write_psf(b0file, b0_status, b0_bundle, b0_legcoeff,
                   b0_xtrace, b0_ytrace, fit_bundles=[0])

        #- bundle 1 (fibers 2,3): every fiber in the bundle failed to fit,
        #- so there are no "selected" fibers for this input file
        b1_status = np.array([NOT_APPLICABLE, NOT_APPLICABLE,
                               FIT_FAILED, FIT_FAILED])
        b1_bundle = np.array([-1, -1, 1, 1])
        b1_legcoeff = np.full((nfibers, ncoeff), 99.0)
        b1_xtrace = np.full((nfibers, ncoeff), 99.0)
        b1_ytrace = np.full((nfibers, ncoeff), 99.0)
        b1file = os.path.join(self.testdir, 'psf-bundle1.fits')
        _write_psf(b1file, b1_status, b1_bundle, b1_legcoeff,
                   b1_xtrace, b1_ytrace)

        outfile = os.path.join(self.testdir, 'psf-merged.fits')
        merge_psf(reffile, [b0file, b1file], outfile)

        with fits.open(outfile) as merged:
            data = merged['PSF'].data
            i_status = np.where(data['PARAM'] == 'STATUS')[0][0]
            merged_status = data['COEFF'][i_status][:, 0]

            #- fibers 0,1 fit successfully
            np.testing.assert_array_equal(merged_status[[0, 1]],
                                           [FIT_OK, FIT_OK])

            #- fibers 2,3 failed to fit; their failure status must still be
            #- recorded, not silently skipped by the `continue` that fires
            #- because bundle 1 has zero selected (status==0) fibers
            np.testing.assert_array_equal(merged_status[[2, 3]],
                                           [FIT_FAILED, FIT_FAILED])

            #- since bundle 1 had no selected fibers, its xtrace/ytrace and
            #- other parameters should NOT have been copied into the output
            i_leg = np.where(data['PARAM'] == 'LEGCOEFF')[0][0]
            merged_legcoeff = data['COEFF'][i_leg]
            np.testing.assert_array_equal(merged_legcoeff[[2, 3]],
                                           ref_legcoeff[[2, 3]])
            np.testing.assert_array_equal(merged['XTRACE'].data[[2, 3]],
                                           ref_xtrace[[2, 3]])
            np.testing.assert_array_equal(merged['YTRACE'].data[[2, 3]],
                                           ref_ytrace[[2, 3]])

            #- bundle 0's successfully fit fibers should be copied over
            np.testing.assert_array_equal(merged_legcoeff[[0, 1]],
                                           b0_legcoeff[[0, 1]])
            np.testing.assert_array_equal(merged['XTRACE'].data[[0, 1]],
                                           b0_xtrace[[0, 1]])
            np.testing.assert_array_equal(merged['YTRACE'].data[[0, 1]],
                                           b0_ytrace[[0, 1]])


def _warning_messages(mock_log):
    """Return the warning messages recorded by a patched get_logger mock"""
    return [str(call.args[0]) for call in mock_log().warning.call_args_list
            if call.args]


class TestMeanPSF(unittest.TestCase):
    """
    Test how mean_psf classifies fiber bundles across input PSFs.

    A bundle can be absent from an input PSF for two very different reasons:
    it was masked out because it sits on a bad CCD amp (every fiber has
    STATUS<0), or its fit genuinely failed (some fiber has STATUS>0). Only the
    second is a fit failure, and two or more of them must abort psfnight.
    These tests cover the four cases seen in production, which are described in
    desihub/desispec#2723:

        20211028  a bad amp in some but not all of the input arcs
        20221121  a bad amp in all of the input arcs
        20250822  crossed fiber traces, i.e. a real fit failure
        20230829  a nominal night

    The fixtures use 4 bundles of 3 fibers and a nominal rchi2 of 1.25, which
    puts mean_psf's selection threshold (median of the non-zero rchi2, plus 1)
    at exactly 2.25 in every test below.
    """

    NBUNDLES = 4
    NFIBERS_PER_BUNDLE = 3
    NOMINAL_RCHI2 = 1.25
    #- one distinct value per input PSF, chosen so that every outcome is
    #- distinguishable: mean of all three is 30, mean of the last two is 40,
    #- and each input on its own is 10, 20 or 60
    VALUES = (10., 20., 60.)

    def setUp(self):
        self.testdir = tempfile.mkdtemp()
        self.outfile = os.path.join(self.testdir, 'psf-mean.fits')
        #- every fiber gets a real bundle id; a -1 here would end up in
        #- np.unique(bundles) and alias the last bundle
        self.bundle = np.repeat(np.arange(self.NBUNDLES),
                                self.NFIBERS_PER_BUNDLE)
        self.nfibers = len(self.bundle)

    def tearDown(self):
        shutil.rmtree(self.testdir, ignore_errors=True)

    def _fibers(self, bundle):
        """Fiber indices belonging to a bundle"""
        return np.where(self.bundle == bundle)[0]

    def _status(self, missing_bundles=(), failed_fibers=(), failed_bundles=()):
        """
        Per-fiber STATUS array: fit ok everywhere except where told otherwise.

        Args:
            missing_bundles: bundles masked out entirely (all fibers STATUS<0)
            failed_fibers: individual fibers whose traces crossed (STATUS=4)
            failed_bundles: bundles where every fiber failed to fit (STATUS=2)
        """
        status = np.full(self.nfibers, FIT_OK)
        for b in missing_bundles:
            status[self._fibers(b)] = NOT_APPLICABLE
        for b in failed_bundles:
            status[self._fibers(b)] = FIT_FAILED
        for fiber in failed_fibers:
            status[fiber] = CROSSED_TRACES
        return status

    def _rchi2(self, **overrides):
        """Nominal rchi2 for every bundle, overridden per bundle by keyword"""
        rchi2 = np.full(self.NBUNDLES, self.NOMINAL_RCHI2)
        for bundle, value in overrides.items():
            rchi2[int(bundle.lstrip('b'))] = value
        return rchi2

    def _write_inputs(self, statuses, rchi2s, **kwargs):
        """Write one input PSF per (status, rchi2) pair; return the filenames"""
        filenames = list()
        for i, (status, rchi2) in enumerate(zip(statuses, rchi2s)):
            filename = os.path.join(self.testdir, 'psf-in-{}.fits'.format(i))
            _write_mean_psf_input(filename, status, self.bundle, rchi2,
                                  self.VALUES[i], **kwargs)
            filenames.append(filename)
        return filenames

    def _read_output(self):
        """Return (dict of PARAM name -> coefficients, per-bundle rchi2)"""
        with fits.open(self.outfile) as hdulist:
            data = hdulist['PSF'].data
            coeff = dict()
            for i, param in enumerate(data['PARAM']):
                coeff[str(param).strip()] = np.array(data['COEFF'][i])
            rchi2 = np.array([hdulist['PSF'].header['B{:02d}RCHI2'.format(b)]
                              for b in range(self.NBUNDLES)])
            xtrace = np.array(hdulist['XTRACE'].data)
        return coeff, rchi2, xtrace

    @patch('desispec.scripts.specex.get_logger')
    def test_bundle_missing_in_some_inputs_averages_over_the_rest(self, mock_log):
        """
        A bundle masked out of some but not all inputs is not a fit failure.

        This is night 20211028, the case desihub/desispec#2723 was opened for:
        bundle 0 is missing from two of the three inputs, which used to look
        like two fit failures and abort psfnight.
        """
        inputs = self._write_inputs(
            [self._status(), self._status(missing_bundles=[0]),
             self._status(missing_bundles=[0])],
            [self._rchi2(), self._rchi2(b0=0.), self._rchi2(b0=0.)])

        mean_psf(inputs, self.outfile)

        coeff, rchi2, _ = self._read_output()
        #- bundle 0 comes from the one input that has it, not from a mean that
        #- would have been dragged towards the two inputs missing it
        np.testing.assert_allclose(coeff['LEGCOEFF'][self._fibers(0)], 10.)
        for b in (1, 2, 3):
            np.testing.assert_allclose(coeff['LEGCOEFF'][self._fibers(b)], 30.)
        #- bundle 0 is still a fit bundle, so its fibers keep STATUS=0
        np.testing.assert_array_equal(coeff['STATUS'][:, 0],
                                      np.zeros(self.nfibers))
        np.testing.assert_allclose(rchi2, self.NOMINAL_RCHI2)

        self.assertTrue(any('present in only 1 of 3' in msg
                            for msg in _warning_messages(mock_log)))
        mock_log().critical.assert_not_called()

    @patch('desispec.scripts.specex.get_logger')
    def test_partially_missing_bundle_is_not_treated_as_missing(self, mock_log):
        """
        A bundle is only "missing" when *every* one of its fibers is masked.

        Real bad-amp exposures have one straddling bundle with a mix of
        STATUS=-1 and STATUS=0 fibers and a perfectly good rchi2 (bundle 10 of
        fit-psf-b8-00154099.fits), which must still be averaged in normally.
        """
        partial = self._status()
        partial[self._fibers(0)[0]] = NOT_APPLICABLE

        inputs = self._write_inputs(
            [partial, self._status(), self._status()],
            [self._rchi2(), self._rchi2(), self._rchi2()])

        mean_psf(inputs, self.outfile)

        coeff, rchi2, _ = self._read_output()
        #- averaged over all three inputs, i.e. the bundle was not dropped
        np.testing.assert_allclose(coeff['LEGCOEFF'][self._fibers(0)], 30.)
        np.testing.assert_allclose(rchi2, self.NOMINAL_RCHI2)
        self.assertFalse(any('present in only' in msg
                             for msg in _warning_messages(mock_log)))
        mock_log().critical.assert_not_called()

    @patch('desispec.scripts.specex.get_logger')
    def test_bundle_missing_in_all_inputs_is_dropped(self, mock_log):
        """
        A bundle masked out of every input is dropped from the merge.

        This is night 20221121, where b8B was missing from all five arcs.
        """
        inputs = self._write_inputs(
            [self._status(missing_bundles=[3])] * 3,
            [self._rchi2(b3=0.)] * 3)

        mean_psf(inputs, self.outfile)

        coeff, rchi2, _ = self._read_output()
        #- the dropped bundle keeps the reference PSF coefficients and is
        #- flagged as not applicable rather than being averaged
        np.testing.assert_allclose(coeff['LEGCOEFF'][self._fibers(3)], 10.)
        np.testing.assert_array_equal(coeff['STATUS'][self._fibers(3), 0],
                                      np.full(self.NFIBERS_PER_BUNDLE,
                                              NOT_APPLICABLE))
        self.assertEqual(rchi2[3], 0.)
        for b in (0, 1, 2):
            np.testing.assert_allclose(coeff['LEGCOEFF'][self._fibers(b)], 30.)
            np.testing.assert_array_equal(coeff['STATUS'][self._fibers(b), 0],
                                          np.zeros(self.NFIBERS_PER_BUNDLE))

        self.assertTrue(any('missing in all input PSFs' in msg
                            for msg in _warning_messages(mock_log)))
        mock_log().critical.assert_not_called()

    def test_two_whole_bundle_fit_failures_raise(self):
        """
        Two inputs whose bundle failed to fit outright must abort the merge.

        Every fiber of the bundle has STATUS>0 and, because merge_psf never
        copies an rchi2 for a bundle with no successfully fit fiber, rchi2=0.
        """
        inputs = self._write_inputs(
            [self._status(failed_bundles=[1]), self._status(failed_bundles=[1]),
             self._status()],
            [self._rchi2(b1=0.), self._rchi2(b1=0.), self._rchi2()])

        with self.assertRaises(RuntimeError) as context:
            mean_psf(inputs, self.outfile)

        message = str(context.exception)
        self.assertIn('2 fit failures', message)
        self.assertIn('bundle 1', message)
        #- the camera is reported without the quotes specex writes into the card
        self.assertIn('camera z7', message)
        self.assertFalse(os.path.exists(self.outfile))

    def test_two_crossed_trace_failures_raise_with_nonzero_rchi2(self):
        """
        Crossed fiber traces must abort the merge even though every rchi2 is fine.

        This is night 20250822, where fibers 250 and 251 of z7 bundle 10 cross.
        The fit converges, so rchi2 is nominal and the old rchi2==0 test could
        not see the failure; only the STATUS>0 written by the specex QA can.
        """
        inputs = self._write_inputs(
            [self._status(failed_fibers=[self._fibers(1)[0],
                                         self._fibers(1)[1]]),
             self._status(failed_fibers=[self._fibers(1)[0]]),
             self._status()],
            [self._rchi2(), self._rchi2(), self._rchi2()])

        with self.assertRaises(RuntimeError) as context:
            mean_psf(inputs, self.outfile)

        message = str(context.exception)
        self.assertIn('2 fit failures', message)
        self.assertIn('bundle 1', message)
        self.assertFalse(os.path.exists(self.outfile))

    @patch('desispec.scripts.specex.get_logger')
    def test_single_fit_failure_warns_without_raising(self, mock_log):
        """One bad input out of three is tolerated, with a warning"""
        inputs = self._write_inputs(
            [self._status(failed_bundles=[1]), self._status(), self._status()],
            [self._rchi2(b1=0.), self._rchi2(), self._rchi2()])

        mean_psf(inputs, self.outfile)

        coeff, rchi2, _ = self._read_output()
        #- the failed input is excluded, leaving the mean of the other two
        np.testing.assert_allclose(coeff['LEGCOEFF'][self._fibers(1)], 40.)
        for b in (0, 2, 3):
            np.testing.assert_allclose(coeff['LEGCOEFF'][self._fibers(b)], 30.)
        np.testing.assert_allclose(rchi2, self.NOMINAL_RCHI2)

        self.assertTrue(any('1 fit failure for bundle 1' in msg
                            for msg in _warning_messages(mock_log)))
        mock_log().critical.assert_not_called()

    def test_fallback_takes_smallest_nonzero_rchi2(self):
        """
        With no acceptable rchi2, the least bad input is used, never a zero one.

        A zero rchi2 means the bundle was not fit at all, so it must lose to a
        merely poor fit.
        """
        inputs = self._write_inputs(
            [self._status(missing_bundles=[1]), self._status(), self._status()],
            [self._rchi2(b1=0.), self._rchi2(b1=5.0), self._rchi2(b1=3.7)])

        mean_psf(inputs, self.outfile)

        coeff, rchi2 = self._read_output()[:2]
        #- input 2 has the smallest non-zero rchi2
        np.testing.assert_allclose(coeff['LEGCOEFF'][self._fibers(1)], 60.)
        self.assertAlmostEqual(rchi2[1], 3.7)

    @patch('desispec.scripts.specex.get_logger')
    def test_fallback_skips_inputs_where_bundle_is_missing(self, mock_log):
        """
        The fallback must not pick an input that lacks the bundle entirely.

        Bundle 1 is masked out of the first two inputs and failed to fit in the
        third, so every rchi2 is zero. Falling back to the first input would
        copy the unfit reference coefficients of an exposure that never had
        this bundle; the one input that does have it has to win.
        """
        inputs = self._write_inputs(
            [self._status(missing_bundles=[1]),
             self._status(missing_bundles=[1]),
             self._status(failed_bundles=[1])],
            [self._rchi2(b1=0.)] * 3)

        mean_psf(inputs, self.outfile)

        coeff, rchi2 = self._read_output()[:2]
        np.testing.assert_allclose(coeff['LEGCOEFF'][self._fibers(1)], 60.)
        self.assertEqual(rchi2[1], 0.)
        #- one real failure among the inputs that have the bundle, not three
        self.assertTrue(any('1 fit failure for bundle 1' in msg
                            for msg in _warning_messages(mock_log)))
        mock_log().critical.assert_not_called()

    def test_single_input_psf(self):
        """A single input PSF is copied through, including an unfit bundle"""
        inputs = self._write_inputs([self._status()], [self._rchi2(b1=0.)])

        mean_psf(inputs, self.outfile)

        coeff, rchi2 = self._read_output()[:2]
        np.testing.assert_allclose(coeff['LEGCOEFF'], 10.)
        self.assertEqual(rchi2[1], 0.)
        for b in (0, 2, 3):
            self.assertAlmostEqual(rchi2[b], self.NOMINAL_RCHI2)

    def test_results_do_not_depend_on_param_row_order(self):
        """
        Bundle classification must not depend on which PARAM row comes first.

        mean_psf loops over PARAM rows to average them; the failure check is
        per bundle, not per row, so it has to give the same answer whether
        STATUS is the first row or the last one as in real specex files.
        """
        orders = [('LEGCOEFF', 'BUNDLE', 'STATUS'),
                  ('STATUS', 'BUNDLE', 'LEGCOEFF')]
        results = list()
        for order in orders:
            with self.subTest(param_order=order):
                subdir = os.path.join(self.testdir, '-'.join(order))
                os.makedirs(subdir)
                self.testdir, parent = subdir, self.testdir
                try:
                    #- tolerated case: same output either way
                    self.outfile = os.path.join(subdir, 'psf-mean.fits')
                    inputs = self._write_inputs(
                        [self._status(), self._status(missing_bundles=[0]),
                         self._status(missing_bundles=[0])],
                        [self._rchi2(), self._rchi2(b0=0.),
                         self._rchi2(b0=0.)],
                        param_order=order)
                    mean_psf(inputs, self.outfile)
                    coeff, rchi2 = self._read_output()[:2]
                    results.append((coeff['LEGCOEFF'], coeff['STATUS'], rchi2))

                    #- failing case: raises either way
                    self.outfile = os.path.join(subdir, 'psf-mean-bad.fits')
                    inputs = self._write_inputs(
                        [self._status(failed_bundles=[1]),
                         self._status(failed_bundles=[1]), self._status()],
                        [self._rchi2(b1=0.), self._rchi2(b1=0.),
                         self._rchi2()],
                        param_order=order)
                    with self.assertRaises(RuntimeError) as context:
                        mean_psf(inputs, self.outfile)
                    self.assertIn('2 fit failures', str(context.exception))
                finally:
                    self.testdir = parent

        for first, second in zip(results[0], results[1]):
            np.testing.assert_allclose(first, second)

    def test_traces_are_averaged_over_all_inputs(self):
        """
        Document that XTRACE/YTRACE are averaged without any bundle selection.

        Unlike the PSF coefficients, the traces are averaged over every input
        regardless of which bundles were masked out or failed, so a bundle
        dropped from the merge still gets a trace contaminated by the exposure
        that was missing it. This is pre-existing behavior, recorded here so
        that a change to it is a deliberate one.
        """
        inputs = self._write_inputs(
            [self._status(missing_bundles=[3])] * 3,
            [self._rchi2(b3=0.)] * 3)

        mean_psf(inputs, self.outfile)

        xtrace = self._read_output()[2]
        np.testing.assert_allclose(xtrace, 30.)


if __name__ == '__main__':
    unittest.main()
