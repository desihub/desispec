"""
Test desispec.scripts.specex
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.polynomial.legendre import legval
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
               rchi2=None, header=None, write_traces=True):
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
        write_traces: if False, omit the XTRACE and YTRACE HDUs entirely, as a
            PSF written by something other than specex might
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

    hdus = [fits.PrimaryHDU(), psf_hdu]
    if write_traces:
        hdus.append(fits.ImageHDU(xtrace, name='XTRACE'))
        hdus.append(fits.ImageHDU(ytrace, name='YTRACE'))

    hdulist = fits.HDUList(hdus)
    hdulist.writeto(filename, overwrite=True)


#- offset added to YTRACE so that a test can tell the two trace HDUs apart and
#- catch an x/y mixup; means are linear, so mean(y) == mean(x) + YTRACE_OFFSET
YTRACE_OFFSET = 1000.


def _legcoeff(filename):
    """Return the LEGCOEFF row of a PSF file, shaped (nfibers, ncoeff)"""
    data = fits.getdata(filename, 'PSF')
    return np.array(data['COEFF'][np.where(data['PARAM'] == 'LEGCOEFF')[0][0]])


def _write_mean_psf_input(filename, status, bundle, rchi2, value, ncoeff=2,
                          param_order=('LEGCOEFF', 'BUNDLE', 'STATUS'),
                          write_traces=True, wavemin=3526.0, wavemax=6055.0):
    """
    Write a merged per-exposure PSF (fit-psf-CAM-EXPID.fits) for mean_psf.

    Args:
        filename: output path
        status: per-fiber STATUS values, 1D array of length nfibers
        bundle: per-fiber BUNDLE values, 1D array of length nfibers
        rchi2: per-bundle rchi2, 0.0 for bundles that are missing or failed
        value: scalar filled into LEGCOEFF/XTRACE so that which input was
            averaged or selected can be read straight off the output. YTRACE
            gets value+YTRACE_OFFSET so the two trace HDUs are distinguishable.
        ncoeff: number of legendre coefficients per fiber
        param_order: see _write_psf; the default puts STATUS last as real
            specex files do
        write_traces: see _write_psf
        wavemin, wavemax: the wavelength range the legendre coefficients are
            defined over. compatible() does not compare these, so inputs may
            disagree and mean_psf then refits them onto the first input's range.

    The header keywords are the ones mean_psf reads directly (PSFVER, CAMERA,
    WAVEMIN, WAVEMAX) plus those compared by specex.compatible(), which must
    match across all inputs or mean_psf calls sys.exit(12).
    """
    nfibers = len(status)
    payload = np.full((nfibers, ncoeff), float(value))
    ypayload = payload + YTRACE_OFFSET
    header = dict(PSFVER='3', CAMERA="'z7      '",
                  WAVEMIN=wavemin, WAVEMAX=wavemax,
                  PSFTYPE='GAUSS-HERMITE', NPIX_X=4096, NPIX_Y=4096,
                  HSIZEX=8, HSIZEY=5, NPARAMS=57, LEGDEG=ncoeff - 1,
                  GHDEGX=6, GHDEGY=6)
    _write_psf(filename, status, bundle, payload, payload, ypayload,
               param_order=param_order, rchi2=rchi2,
               header=header, write_traces=write_traces)


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

        #- bundle 0 (fibers 0,1): fit succeeded.  Note that specex fills in
        #- BUNDLE for every fiber of a per-bundle file, not just the fibers of
        #- the bundle it fit, so out-of-bundle fibers keep their real bundle id
        #- and are distinguished only by STATUS.
        b0_status = np.array([FIT_OK, FIT_OK, NOT_APPLICABLE, NOT_APPLICABLE])
        b0_bundle = np.array([0, 0, 1, 1])
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
        b1_bundle = np.array([0, 0, 1, 1])
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

    def _write_inputs(self, statuses, rchi2s, no_traces=(), wave_ranges=None,
                      **kwargs):
        """
        Write one input PSF per (status, rchi2) pair; return the filenames

        Args:
            statuses, rchi2s: one per input, see _status and _rchi2
            no_traces: indices of inputs to write without XTRACE/YTRACE HDUs
            wave_ranges: optional (wavemin, wavemax) per input, to make some of
                them disagree about the range their coefficients cover
            kwargs: passed to _write_mean_psf_input for every input
        """
        filenames = list()
        for i, (status, rchi2) in enumerate(zip(statuses, rchi2s)):
            filename = os.path.join(self.testdir, 'psf-in-{}.fits'.format(i))
            wave = dict()
            if wave_ranges is not None:
                wave = dict(zip(('wavemin', 'wavemax'), wave_ranges[i]))
            _write_mean_psf_input(filename, status, self.bundle, rchi2,
                                  self.VALUES[i],
                                  write_traces=(i not in no_traces),
                                  **wave, **kwargs)
            filenames.append(filename)
        return filenames

    def _read_output(self):
        """
        Return the merged output

        Returns:
            (coeff, rchi2, xtrace, ytrace) where coeff maps a PARAM name to its
            coefficients and rchi2 is the per-bundle B{bb}RCHI2 of the header
        """
        with fits.open(self.outfile) as hdulist:
            data = hdulist['PSF'].data
            coeff = dict()
            for i, param in enumerate(data['PARAM']):
                coeff[str(param).strip()] = np.array(data['COEFF'][i])
            rchi2 = np.array([hdulist['PSF'].header['B{:02d}RCHI2'.format(b)]
                              for b in range(self.NBUNDLES)])
            xtrace = np.array(hdulist['XTRACE'].data)
            ytrace = np.array(hdulist['YTRACE'].data)
        return coeff, rchi2, xtrace, ytrace

    def _assert_traces(self, expected_per_bundle):
        """
        Assert the merged XTRACE/YTRACE of each bundle

        Args:
            expected_per_bundle: dict of bundle id -> expected XTRACE value.
                YTRACE is checked at that value plus YTRACE_OFFSET, so an x/y
                mixup in mean_psf cannot pass.
        """
        xtrace, ytrace = self._read_output()[2:]
        for bundle, expected in expected_per_bundle.items():
            fibers = self._fibers(bundle)
            np.testing.assert_allclose(xtrace[fibers], expected,
                err_msg='XTRACE of bundle {}'.format(bundle))
            np.testing.assert_allclose(ytrace[fibers], expected + YTRACE_OFFSET,
                err_msg='YTRACE of bundle {}'.format(bundle))

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

        coeff, rchi2 = self._read_output()[:2]
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

        coeff, rchi2 = self._read_output()[:2]
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

        coeff, rchi2 = self._read_output()[:2]
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

        coeff, rchi2 = self._read_output()[:2]
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

    #- The tests below cover desihub/desispec#2819: XTRACE/YTRACE used to be
    #- averaged with a bare np.mean over every input, with none of the
    #- per-bundle selection applied to the PSF coefficients, so a bundle that
    #- some arcs never fit still had their unfitted reference traces blended
    #- into its own. specex fits the traces as free parameters of the same
    #- per-bundle fit that produces the rchi2, so the two must agree.

    def test_traces_are_averaged_over_all_inputs_on_a_nominal_night(self):
        """
        With nothing wrong, the traces are the mean of every input as before.

        This is night 20230829, and it is the case that must not change: the
        per-bundle selection keeps all of the inputs, so psfnight for a normal
        night comes out exactly as it did before #2819 was fixed.
        """
        inputs = self._write_inputs([self._status()] * 3,
                                    [self._rchi2()] * 3)

        mean_psf(inputs, self.outfile)

        self._assert_traces({b: 30. for b in range(self.NBUNDLES)})

    def test_traces_use_the_same_bundle_selection_as_the_coefficients(self):
        """
        A bundle masked out of some inputs takes its traces from the rest.

        This is night 20211028. Bundle 0 was fit by one arc only, so both its
        coefficients and its traces must come from that arc; averaging in the
        two arcs that never fit it is what #2819 reported.
        """
        inputs = self._write_inputs(
            [self._status(), self._status(missing_bundles=[0]),
             self._status(missing_bundles=[0])],
            [self._rchi2(), self._rchi2(b0=0.), self._rchi2(b0=0.)])

        mean_psf(inputs, self.outfile)

        self._assert_traces({0: 10., 1: 30., 2: 30., 3: 30.})
        #- the traces now agree with the coefficients for every bundle
        coeff = self._read_output()[0]
        np.testing.assert_allclose(coeff['LEGCOEFF'][self._fibers(0)], 10.)

    def test_dropped_bundle_keeps_the_reference_traces(self):
        """
        A bundle masked out of every input keeps the reference PSF's traces.

        This is night 20221121, where b8B was missing from all five arcs. The
        coefficients of such a bundle already fall back to the reference PSF,
        and the traces have to do the same rather than average three exposures
        that never fit it.
        """
        inputs = self._write_inputs(
            [self._status(missing_bundles=[3])] * 3,
            [self._rchi2(b3=0.)] * 3)

        mean_psf(inputs, self.outfile)

        #- 10. is the reference, i.e. the first input, not the mean of 30.
        self._assert_traces({0: 30., 1: 30., 2: 30., 3: 10.})

    def test_traces_exclude_a_bundle_that_failed_to_fit(self):
        """A bundle that failed to fit in one arc is excluded from its traces"""
        inputs = self._write_inputs(
            [self._status(failed_bundles=[1]), self._status(), self._status()],
            [self._rchi2(b1=0.), self._rchi2(), self._rchi2()])

        mean_psf(inputs, self.outfile)

        #- mean of the two good inputs, 20. and 60.
        self._assert_traces({0: 30., 1: 40., 2: 30., 3: 30.})

    def test_traces_reject_an_input_above_the_rchi2_threshold(self):
        """
        An arc rejected by the rchi2 cut is excluded from the traces too.

        This is the case that distinguishes reusing the whole coefficient
        selection from merely skipping bundles nothing fit. specex fits the
        traces in the same least-squares whose final chi2 becomes B{bb}RCHI2
        (the fit_trace stages of FitEverything), so a bundle whose rchi2 says
        the model fit the arc badly has a suspect trace solution as well.
        """
        inputs = self._write_inputs(
            [self._status()] * 3,
            [self._rchi2(), self._rchi2(b1=5.0), self._rchi2()])

        mean_psf(inputs, self.outfile)

        #- 5.0 is above the 2.25 threshold, leaving the mean of 10. and 60.
        self._assert_traces({0: 30., 1: 35., 2: 30., 3: 30.})
        coeff, rchi2 = self._read_output()[:2]
        np.testing.assert_allclose(coeff['LEGCOEFF'][self._fibers(1)], 35.)
        self.assertAlmostEqual(rchi2[1], self.NOMINAL_RCHI2)

    def test_traces_average_present_inputs_when_no_rchi2_passes(self):
        """
        With no acceptable rchi2 the traces average instead of picking one.

        Bundle 1 is masked out of the first input and fit badly by the other
        two. The coefficients have to choose an input and take the least bad
        one, but rchi2_threshold is a relative cut: when nothing passes it,
        nothing is an outlier and the ranking among near-equal bad values is
        noise. The traces therefore keep averaging over the inputs that have
        the bundle rather than inheriting that arbitrary choice, which is the
        one place the traces deliberately diverge from the coefficients.

        Measured on real arcs, this is the common case: on loa nights 20230829,
        20211028 and 20221121 about 15 of 600 camera-bundles have no input
        passing the cut, against about 10 where the cut rejects a real outlier.
        """
        inputs = self._write_inputs(
            [self._status(missing_bundles=[1]), self._status(), self._status()],
            [self._rchi2(b1=0.), self._rchi2(b1=5.0), self._rchi2(b1=3.7)])

        mean_psf(inputs, self.outfile)

        #- mean of inputs 1 and 2, the ones that have the bundle
        self._assert_traces({0: 30., 1: 40., 2: 30., 3: 30.})
        #- while the coefficients still take input 2, the smallest non-zero
        coeff = self._read_output()[0]
        np.testing.assert_allclose(coeff['LEGCOEFF'][self._fibers(1)], 60.)

    def test_traces_average_all_inputs_when_every_rchi2_is_equally_bad(self):
        """
        A bundle every arc fits badly keeps its traces averaged over them all.

        This is b8 bundle 10 of night 20221121, where the five arcs fit at
        rchi2 4.95, 4.62, 4.81, 5.03 and 4.61: all bad, none an outlier. The
        coefficients take the 4.61 arc, but preferring it for the traces over
        the 5.03 one would trade the averaging of five exposures for an 8%
        difference in rchi2 that carries no real information.
        """
        inputs = self._write_inputs(
            [self._status()] * 3,
            [self._rchi2(b1=4.9), self._rchi2(b1=5.0), self._rchi2(b1=4.6)])

        mean_psf(inputs, self.outfile)

        #- unchanged from the pre-#2819 behavior for this bundle
        self._assert_traces({b: 30. for b in range(self.NBUNDLES)})
        coeff = self._read_output()[0]
        np.testing.assert_allclose(coeff['LEGCOEFF'][self._fibers(1)], 60.)

    def test_inputs_on_another_wavelength_range_are_refit_before_averaging(self):
        """
        Coefficients and traces are refit when an input uses another range.

        compatible() does not compare WAVEMIN/WAVEMAX, so mean_psf accepts
        inputs whose legendre coefficients cover a different wavelength range
        and re-expresses them on the first input's range before averaging.

        The traces are left non-constant on purpose. A constant is the same
        constant in any parametrization, so it would refit to itself and this
        test would pass even if the refit never ran. Checking instead that the
        merged trace evaluated at a wavelength equals the mean of the inputs
        evaluated at that same wavelength tests the refit for real, and does so
        without reimplementing its algebra here.

        This path had no coverage at all, which is how the reference to the
        unrelated icoeff of the coefficient loop survived in it for so long.
        """
        ranges = [(3526.0, 6055.0), (3600.0, 5900.0), (3526.0, 6055.0)]
        inputs = self._write_inputs(
            [self._status()] * 3, [self._rchi2()] * 3,
            #- the first input defines the output range
            wave_ranges=ranges)

        mean_psf(inputs, self.outfile)

        #- sample the merged traces and the inputs on the same wavelengths
        wavemin, wavemax = ranges[0]
        wave = np.linspace(wavemin, wavemax, 9)

        def evaluate(coefficients, wave_range):
            """Legendre series of each fiber, evaluated at wave"""
            lo, hi = wave_range
            u = (wave - lo) / (hi - lo) * 2. - 1.
            return np.array([legval(u, c) for c in coefficients])

        expected_x = np.mean(
            [evaluate(fits.getdata(f, 'XTRACE'), r)
             for f, r in zip(inputs, ranges)], axis=0)
        expected_y = np.mean(
            [evaluate(fits.getdata(f, 'YTRACE'), r)
             for f, r in zip(inputs, ranges)], axis=0)
        expected_c = np.mean(
            [evaluate(_legcoeff(f), r) for f, r in zip(inputs, ranges)], axis=0)

        coeff, _, xtrace, ytrace = self._read_output()
        np.testing.assert_allclose(evaluate(xtrace, ranges[0]), expected_x,
                                   atol=1e-8)
        np.testing.assert_allclose(evaluate(ytrace, ranges[0]), expected_y,
                                   atol=1e-8)
        np.testing.assert_allclose(evaluate(coeff['LEGCOEFF'], ranges[0]),
                                   expected_c, atol=1e-8)
        #- and the refit really was needed, i.e. input 1 was not already equal
        self.assertFalse(np.allclose(
            evaluate(fits.getdata(inputs[1], 'XTRACE'), ranges[0]),
            evaluate(fits.getdata(inputs[1], 'XTRACE'), ranges[1])))

    @patch('desispec.scripts.specex.get_logger')
    def test_inputs_without_trace_hdus_are_dropped_per_bundle(self, mock_log):
        """
        An input lacking trace HDUs is left out without disabling selection.

        XTRACE/YTRACE are only read where the HDUs exist, but the None
        placeholders keep the trace lists indexed by input, so such an input can
        simply be dropped from each bundle's selection. Disabling selection for
        the whole camera instead would reintroduce #2819 for bundles whose
        correct selection is perfectly well known, which is what input 1 would
        otherwise do to bundle 0 here.
        """
        inputs = self._write_inputs(
            [self._status(), self._status(missing_bundles=[0]),
             self._status(missing_bundles=[0])],
            [self._rchi2(), self._rchi2(b0=0.), self._rchi2(b0=0.)],
            no_traces=(1,))

        mean_psf(inputs, self.outfile)

        #- bundle 0 was fit only by input 0, which does have traces, so it is
        #- used alone rather than averaged with input 2's reference trace
        self._assert_traces({0: 10., 1: 35., 2: 35., 3: 35.})
        #- and the traces now agree with the coefficients for that bundle
        coeff = self._read_output()[0]
        np.testing.assert_allclose(coeff['LEGCOEFF'][self._fibers(0)], 10.)
        self.assertTrue(any('have both' in msg
                            for msg in _warning_messages(mock_log)))

    @patch('desispec.scripts.specex.get_logger')
    def test_fallback_traces_exclude_inputs_that_never_fit_the_bundle(self, mock_log):
        """
        The averaging fallback uses fitted inputs, not merely present ones.

        A bundle that failed outright is still "present": every fiber has
        STATUS>0 rather than <0. But merge_psf only copies XTRACE/YTRACE for
        STATUS==0 fibers, so such an input carries the reference traces, and
        averaging it in is exactly the contamination #2819 is about. Input 0
        here failed the bundle outright while 1 and 2 fit it but land above the
        cut, so the traces must average 1 and 2 only.
        """
        inputs = self._write_inputs(
            [self._status(failed_bundles=[1]), self._status(), self._status()],
            [self._rchi2(b1=0.), self._rchi2(b1=5.0), self._rchi2(b1=3.7)])

        mean_psf(inputs, self.outfile)

        #- mean of 20 and 60, not of 10, 20 and 60
        self._assert_traces({0: 30., 1: 40., 2: 30., 3: 30.})
        #- the coefficients still take the smallest non-zero rchi2, input 2
        coeff = self._read_output()[0]
        np.testing.assert_allclose(coeff['LEGCOEFF'][self._fibers(1)], 60.)
        mock_log().critical.assert_not_called()

    def test_fallback_traces_drop_an_outlier_among_the_fitted_inputs(self):
        """
        The fallback compares inputs against this bundle's own rchi2.

        rchi2_threshold is built from the median over every bundle of the
        camera, so a bundle can sit entirely above it while still containing a
        clear outlier. Averaging is only defensible where the inputs really are
        comparable, so the fallback applies the same median+1 rule to the
        bundle's own values, which keeps near-equal inputs and drops a lone bad
        one instead of blending it in.
        """
        inputs = self._write_inputs(
            [self._status()] * 3,
            [self._rchi2(b1=4.6), self._rchi2(b1=5.0), self._rchi2(b1=100.)])

        mean_psf(inputs, self.outfile)

        #- bundle 1's own threshold is median(4.6, 5.0, 100) + 1 = 6.0, so the
        #- rchi2=100 input is excluded and the other two are averaged
        self._assert_traces({0: 30., 1: 15., 2: 30., 3: 30.})


if __name__ == '__main__':
    unittest.main()
