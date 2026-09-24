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
from desispec.trace_shifts import compute_dx_from_cross_dispersion_profiles, legx

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
