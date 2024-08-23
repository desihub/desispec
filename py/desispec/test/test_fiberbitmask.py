"""
test desispec.fiberflat
"""

import unittest
import copy
import os
from uuid import uuid1

import numpy as np

from desispec.test.util import get_frame_data
from desispec.fiberbitmasking import get_fiberbitmasked_frame_arrays
from desispec.fiberbitmasking import get_fiberbitmask_comparison_value
from desispec.maskbits import fibermask

class TestFrameBitMask(unittest.TestCase):


    def setUp(self):
        self.frame = get_frame_data(10)

    def tearDown(self):
        pass

    def test_framebitmask(self):
        self.frame.fibermap['FIBERSTATUS'][1] = 1
        self.assertTrue( np.any(self.frame.ivar[0] != 0) )
        self.assertTrue( np.any(self.frame.ivar[1] != 0) )

        ivar1 = get_fiberbitmasked_frame_arrays(self.frame, bitmask=1)
        self.assertTrue( np.any(ivar1[0] != 0) )  #- unchanged
        self.assertTrue( np.all(ivar1[1] == 0) )  #- masked

        #- offset fiber numbers and repeat
        self.frame.fibermap['FIBER'] += 15

        self.assertTrue( np.any(self.frame.ivar[0] != 0) )
        self.assertTrue( np.any(self.frame.ivar[1] != 0) )

        ivar2 = get_fiberbitmasked_frame_arrays(self.frame, bitmask=1)
        self.assertTrue( np.all(ivar1 == ivar2) )

    def check_mask(self, bitname, ok_steps, bad_steps):
        """
        Check get_fiberbitmask_comparison_value(step, 'b') for every step.
        FIBERSTATUS fibermask.mask(bitname) should be set for every step
        in bad_steps and not set for every step in ok_steps.
        """
        for step in ok_steps:
            mask = get_fiberbitmask_comparison_value(step, 'b')
            self.assertTrue(mask & fibermask.mask(bitname) == 0, f"{step=} unnecessarily excludes {bitname}")

        for step in bad_steps:
            mask = get_fiberbitmask_comparison_value(step, 'b')
            self.assertTrue(mask & fibermask.mask(bitname) != 0, f"{step=} should exclude {bitname} but doesn't")

    def test_ambiguous_maskbits(self):
        """Test cases that are bad for some steps but not for others
        """

        # NOTE: fiberfitmask doesn't currently support arc

        #- BROKENFIBER is bad for everything
        self.check_mask('BROKENFIBER', ok_steps=[], bad_steps=['flat', 'sky', 'stdstar', 'fluxcalib'])

        #- RESTRICTED is ok for everything
        self.check_mask('RESTRICTED', ok_steps=['flat', 'sky', 'stdstar', 'fluxcalib'], bad_steps=[])

        #- BADPOSITION is ok for flats, but bad for others
        self.check_mask('BADPOSITION', ok_steps=['flat',], bad_steps=['sky', 'stdstar', 'fluxcalib'])

        #- POORPOSITION is ok for flats, sky, and fluxcalib; but bad for stdstars
        self.check_mask('POORPOSITION', ok_steps=['flat', 'sky', 'fluxcalib'], bad_steps=['stdstar'])

        #- NEARCHARGETRAP and VARIABLETHRU are informative for fiberbitmasking;
        #- i.e. they don't trigger masking fibers
        #- TODO: it's actually bad for faint targets and sky for a single amp, but we structurally
        #- don't have a way to encode that in FIBERSTATUS (fiber not CCD or amp)
        self.check_mask('NEARCHARGETRAP', ok_steps=['flat', 'sky', 'stdstar', 'fluxcalib'], bad_steps=[])
        self.check_mask('VARIABLETHRU', ok_steps=['flat', 'sky', 'stdstar', 'fluxcalib'], bad_steps=[])

