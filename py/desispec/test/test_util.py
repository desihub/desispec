"""
Test desispec.util.*
"""

import unittest
import numpy as np

from desispec import util

class TestNight(unittest.TestCase):
    
    def test_night(self):
        self.assertEqual(util.ymd2night(2015, 1, 2), '20150102')
        self.assertEqual(util.night2ymd('20150102'), (2015, 1, 2))
        self.assertRaises(ValueError, util.night2ymd, '20150002')
        self.assertRaises(ValueError, util.night2ymd, '20150100')
        self.assertRaises(ValueError, util.night2ymd, '20150132')
        self.assertRaises(ValueError, util.night2ymd, '20151302')

    def test_mask32(self):
        for dtype in (
            int, 'int64', 'uint64', 'i8', 'u8',
            'int32', 'uint32', 'i4', 'u4',
            'int16', 'uint16', 'i2', 'u2',
            'int8', 'uint8', 'i1', 'u1',
            ):
            x = np.ones(10, dtype=np.dtype(dtype))
            m32 = util.mask32(x)                
            self.assertTrue(np.all(m32 == 1))
            
        x = util.mask32( np.array([-1,0,1], dtype='i4') )
        self.assertEqual(x[0], 2**32-1)
        self.assertEqual(x[1], 0)
        self.assertEqual(x[2], 1)
        
        with self.assertRaises(ValueError):
            util.mask32(np.arange(2**35, 2**35+5))

        with self.assertRaises(ValueError):
            util.mask32(np.arange(-2**35, -2**35+5))

    def test_combine_ivar(self):
        #- input inverse variances with some zeros (1D)
        ivar1 = np.random.uniform(-1, 10, size=200).clip(0)
        ivar2 = np.random.uniform(-1, 10, size=200).clip(0)
        ivar = util.combine_ivar(ivar1, ivar2)
        izero = np.where(ivar1 == 0)
        self.assertTrue(np.all(ivar[izero] == 0))
        izero = np.where(ivar2 == 0)
        self.assertTrue(np.all(ivar[izero] == 0))
        self.assertTrue(ivar.dtype == np.float64)
        
        #- input inverse variances with some zeros (2D)
        np.random.seed(0)
        ivar1 = np.random.uniform(-1, 10, size=(10,20)).clip(0)
        ivar2 = np.random.uniform(-1, 10, size=(10,20)).clip(0)
        ivar = util.combine_ivar(ivar1, ivar2)
        izero = np.where(ivar1 == 0)
        self.assertTrue(np.all(ivar[izero] == 0))
        izero = np.where(ivar2 == 0)
        self.assertTrue(np.all(ivar[izero] == 0))
        self.assertTrue(ivar.dtype == np.float64)

        #- Dimensionality
        self.assertRaises(AssertionError, util.combine_ivar, ivar1, ivar2[0])

        #- ivar must be positive
        self.assertRaises(AssertionError, util.combine_ivar, -ivar1, ivar2)
        self.assertRaises(AssertionError, util.combine_ivar, ivar1, -ivar2)
        
        #- does it actually combine them correctly?
        ivar = util.combine_ivar(1, 2)
        self.assertEqual(ivar, 1.0/(1.0 + 0.5))
        
        #- float -> float, int -> float, 0-dim ndarray -> 0-dim ndarray
        ivar = util.combine_ivar(1, 2)
        self.assertTrue(isinstance(ivar, float))
        ivar = util.combine_ivar(1.0, 2.0)
        self.assertTrue(isinstance(ivar, float))
        ivar = util.combine_ivar(np.asarray(1.0), np.asarray(2.0))
        self.assertTrue(isinstance(ivar, np.ndarray))
        self.assertEqual(ivar.ndim, 0)
        
        
if __name__ == '__main__':
    unittest.main()
        