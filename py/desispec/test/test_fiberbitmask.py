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

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
