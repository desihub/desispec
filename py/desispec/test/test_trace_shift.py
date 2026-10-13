"""
Test code related to trace_shifts

This tests the spot-finding bug at the edge of images from issue #2634.
Otherwise it is a placeholder for adding future tests with more coverage
of the trace shift fitting code.
"""

import unittest
import numpy as np
from numpy.polynomial.legendre import legval
from desispec.large_trace_shifts import detect_spots_in_image
from desispec.trace_shifts import compute_dx_from_cross_dispersion_profiles, legx, _continuum_subtract_median, _weighted_median

# dummy class to mimic Image object
class ImageLite:
    def __init__(self, pix):
        self.pix = pix
        self.ivar = np.ones_like(pix)
        self.mask = np.zeros_like(pix, dtype=np.uint32)

class TestTraceShift(unittest.TestCase):
    def test_detect_spots_in_image(self):
        # Create a dummy image with some bright spots
        rnd = np.random.RandomState(0)
        pix = rnd.normal(size=(100, 100))
        pix[20:25, 30:35] = 100
        pix[50:55, 50:55] = 150
        pix[80:85, 70:75] = 200
        pix[0:5, 70:75] = 100    # note on edge

        image = ImageLite(pix)

        # Detect spots in the image
        xc, yc = detect_spots_in_image(image)

        self.assertEqual(len(xc), 4)
        self.assertEqual(len(yc), 4)

        self.assertFalse(np.any(np.isnan(xc)), 'xc has NaN values')
        self.assertFalse(np.any(np.isnan(yc)), 'yc has NaN values')

    def test_compute_dx_from_cross_dispersion_profiles(self):
        """Test that returned (x, y, wave) lie on the input trace for any image_rebin (issue #2849)"""
        nrows, ncols = 1000, 50
        wavemin, wavemax = 5000.0, 6000.0

        # linear traces: y = row of wave, x tilted by 0.02 pixel per row;
        # pixel centers are at integer coordinates
        ycoef = np.array([[(nrows - 1) / 2, (nrows - 1) / 2], ])
        xcoef = np.array([[10 + 0.02 * (nrows - 1) / 2, 0.02 * (nrows - 1) / 2], ])

        def wave_of_y(y):
            return wavemin + (wavemax - wavemin) * y / (nrows - 1)

        # Gaussian cross-dispersion profile offset by true_dx from the trace
        true_dx = 0.3
        yy, xx = np.mgrid[0:nrows, 0:ncols]
        xtrace = legval(legx(wave_of_y(yy), wavemin, wavemax), xcoef[0])
        pix = 1000 * np.exp(-0.5 * ((xx - xtrace - true_dx) / 0.7)**2)
        image = ImageLite(pix)

        for image_rebin in (1, 2, 4):
            x, y, dx, ex, fiber, wave = compute_dx_from_cross_dispersion_profiles(
                xcoef, ycoef, wavemin, wavemax, image, image_rebin=image_rebin)

            self.assertGreater(len(y), 5)
            np.testing.assert_allclose(dx, true_dx, atol=0.01, err_msg=f'image_rebin={image_rebin}')

            # returned coordinates must be consistent with the trace
            rwave = legx(wave, wavemin, wavemax)
            np.testing.assert_allclose(y, legval(rwave, ycoef[0]), atol=0.01,
                                       err_msg=f'y vs. wave image_rebin={image_rebin}')
            np.testing.assert_allclose(x, legval(rwave, xcoef[0]), atol=0.01,
                                       err_msg=f'x vs. wave image_rebin={image_rebin}')

    def test_continuum_subtract_median_masked_amp(self):
        """Test that pixels masked in some fibers (e.g. one masked amp) do not bias the median"""
        nfibers, npix = 100, 400
        rnd = np.random.RandomState(0)
        lines = np.zeros(npix)
        lines[50::50] = 100.
        flux = 10 + lines[None, :] + rnd.normal(size=(nfibers, npix))
        ivar = np.ones((nfibers, npix))

        mflux0, mivar0, _ = _continuum_subtract_median(flux, ivar)

        # mask half the fibers over the second half of the spectrum,
        # like a masked amplifier, plus a few more over a short range
        flux[50:, 200:] = 0.
        ivar[50:, 200:] = 0.
        flux[:45, 300:320] = 0.
        ivar[:45, 300:320] = 0.

        mflux, mivar, _ = _continuum_subtract_median(flux, ivar)

        # lines in the masked half are recovered from the unmasked fibers
        np.testing.assert_allclose(mflux[250::50], mflux0[250::50], rtol=0.05)
        self.assertTrue(np.all(mivar[200:300] > 0))

        # pixels with fewer than min_fibers good fibers get no weight
        self.assertTrue(np.all(mivar[300:320] == 0))
        self.assertTrue(np.all(mivar[320:] > 0))

    def test_weighted_median(self):
        """Test the weighted median along the fiber axis"""
        rnd = np.random.RandomState(0)
        x = rnd.normal(size=(11, 5))

        # equal weights give the ordinary median
        np.testing.assert_allclose(_weighted_median(x, np.ones_like(x)), np.median(x, axis=0))

        # zero-weight entries are ignored
        w = np.ones_like(x)
        w[:4] = 0
        np.testing.assert_allclose(_weighted_median(x, w), np.median(x[4:], axis=0))

        # a dominant weight selects that entry
        w = np.ones_like(x)
        w[3] = 100
        np.testing.assert_allclose(_weighted_median(x, w), x[3])

        # no weight at all gives 0
        np.testing.assert_allclose(_weighted_median(x, np.zeros_like(x)), 0)

    def test_continuum_subtract_median_bright_stars(self):
        """Test that sky lines on top of stellar absorption in most fibers are recovered (issue #2891)"""
        nfibers, npix, nsky = 100, 400, 30
        rnd = np.random.RandomState(0)
        x = np.arange(npix)
        centers = (100, 200, 300)
        sky = 20 + sum(50 * np.exp(-0.5 * ((x - c) / 1.5)**2) for c in centers)
        star = 500 * (1 - sum(0.3 * np.exp(-0.5 * ((x - c - 1) / 2.)**2) for c in centers))

        # first nsky fibers are sky fibers, the others bright stars (like on backup tiles);
        # same noise for all fibers so that only the contamination term distinguishes them
        is_sky_fiber = np.arange(nfibers) < nsky
        model = sky[None, :] + (~is_sky_fiber)[:, None] * star[None, :]
        flux = model + rnd.normal(size=model.shape) * 3.
        ivar = np.ones_like(model) / 3.**2

        mflux_sky, _, _ = _continuum_subtract_median(flux[is_sky_fiber], ivar[is_sky_fiber])
        mflux_plain, _, _ = _continuum_subtract_median(flux, ivar)
        peaks = np.array(centers)

        # the plain median is dominated by the stellar absorption and erases the sky lines
        self.assertTrue(np.all(mflux_plain[peaks] < 0.2 * mflux_sky[peaks]))

        # the weighted median recovers them, with or without sky fibers identified
        for is_sky in (is_sky_fiber, None):
            mflux, mivar, _ = _continuum_subtract_median(flux, ivar, contamination_frac=0.3, is_sky_fiber=is_sky)
            np.testing.assert_allclose(mflux[peaks], mflux_sky[peaks], rtol=0.1)
            self.assertTrue(np.all(mivar[peaks] > 0))

        # without bright objects the weighted median matches the plain median
        flux_sky, ivar_sky = flux[is_sky_fiber], ivar[is_sky_fiber]
        mflux, _, _ = _continuum_subtract_median(flux_sky, ivar_sky, contamination_frac=0.3,
                                                 is_sky_fiber=np.ones(nsky, dtype=bool))
        np.testing.assert_allclose(mflux[peaks], mflux_sky[peaks], rtol=0.05)
