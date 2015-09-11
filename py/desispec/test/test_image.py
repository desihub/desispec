import unittest

import numpy as np
from desispec.image import Image

# Image(self, pix, ivar, mask=None, readnoise=0.0, camera='unknown'):
class TestImage(unittest.TestCase):

    def setUp(self):
        shape = (5,5)
        self.pix = np.random.uniform(size=shape)
        self.ivar = np.random.uniform(size=shape)
        self.ivar[0] = 0.0
        self.mask = np.random.randint(0, 3, size=shape)

    def test_init(self):
        image = Image(self.pix, self.ivar)
        self.assertTrue(np.all(image.pix == self.pix))
        self.assertTrue(np.all(image.ivar == self.ivar))
        
    def test_mask(self):
        image = Image(self.pix, self.ivar)
        self.assertTrue(np.all(image.mask == (self.ivar==0)))

        image = Image(self.pix, self.ivar, self.mask)
        self.assertTrue(np.all(image.mask == self.mask))

    def test_readnoise(self):
        image = Image(self.pix, self.ivar)
        self.assertEqual(image.readnoise, 0.0)
        image = Image(self.pix, self.ivar, readnoise=1.0)
        self.assertEqual(image.readnoise, 1.0)

    def test_camera(self):
        image = Image(self.pix, self.ivar)
        self.assertEqual(image.camera, 'unknown')
        image = Image(self.pix, self.ivar, camera='b0')
        self.assertEqual(image.camera, 'b0')
        
    def test_assertions(self):
        with self.assertRaises(ValueError):
            Image(self.pix[0], self.ivar[0])    #- pix not 2D
        with self.assertRaises(ValueError):
            Image(self.pix, self.ivar[0])       #- pix.shape != ivar.shape
        with self.assertRaises(ValueError):
            Image(self.pix, self.ivar, self.mask[:, 0:1])   #- pix.shape != mask.shape

        
#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
