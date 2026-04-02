"""
Test code related to trace_shifts

This tests the spot-finding bug at the edge of images from issue #2634.
Otherwise it is a placeholder for adding future tests with more coverage
of the trace shift fitting code.
"""

import unittest
import numpy as np
from desispec.large_trace_shifts import detect_spots_in_image

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
