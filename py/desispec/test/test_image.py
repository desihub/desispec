import unittest

import numpy as np
from desispec.image import Image
from desispec.maskbits import ccdmask

# Image(self, pix, ivar, mask=None, readnoise=0.0, camera='unknown'):
class TestImage(unittest.TestCase):

    def setUp(self):
        shape = (5,5)
        self.pix = np.random.uniform(size=shape)
        self.ivar = np.random.uniform(size=shape)
        self.ivar[0] = 0.0
        self.mask = np.random.randint(0, 3, size=shape).astype(np.uint32)

    def test_init(self):
        image = Image(self.pix, self.ivar)
        self.assertTrue(np.all(image.pix == self.pix))
        self.assertTrue(np.all(image.ivar == self.ivar))
        
    def test_mask(self):
        image = Image(self.pix, self.ivar)
        self.assertTrue(np.all(image.mask == (self.ivar==0)*ccdmask.BAD))
        self.assertEqual(image.mask.dtype, np.uint32)

        image = Image(self.pix, self.ivar, self.mask)
        self.assertTrue(np.all(image.mask == self.mask))
        self.assertEqual(image.mask.dtype, np.uint32)

        image = Image(self.pix, self.ivar, self.mask.astype(np.int16))
        self.assertEqual(image.mask.dtype, np.uint32)

    def test_readnoise(self):
        image = Image(self.pix, self.ivar)
        self.assertEqual(image.readnoise, 0.0)
        image = Image(self.pix, self.ivar, readnoise=1.0)
        self.assertEqual(image.readnoise, 1.0)
        readnoise = np.random.uniform(size=self.pix.shape)
        image = Image(self.pix, self.ivar, readnoise=readnoise)
        self.assertTrue(np.all(image.readnoise == readnoise))

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
            
    def test_slicing(self):
        meta = dict(blat='foo', NAXIS1=self.pix.shape[1], NAXIS2=self.pix.shape[0])
        img1 = Image(self.pix, self.ivar, meta=meta)
        nx, ny = 2, 3
        img2 = img1[0:ny, 0:nx]
        self.assertEqual(img2.pix.shape[0], ny)
        self.assertEqual(img2.pix.shape[1], nx)
        self.assertTrue(img2.pix.shape == img2.ivar.shape)
        self.assertTrue(img2.pix.shape == img2.mask.shape)

        self.assertEqual(img1.camera, img2.camera)
        self.assertEqual(img1.readnoise, img2.readnoise)
        for key in img1.meta:
            if key != 'NAXIS1' and key != 'NAXIS2':
                self.assertEqual(img1.meta[key], img2.meta[key])
        self.assertFalse(img1.meta is img2.meta)

        self.assertEqual(img2.meta['NAXIS1'], nx)
        self.assertEqual(img2.meta['NAXIS2'], ny)
        
        #- also works for non-None mask and meta=None
        img1 = Image(self.pix, self.ivar, mask=(self.ivar==0))
        nx, ny = 2, 3
        img2 = img1[0:ny, 0:nx]
        self.assertEqual(img2.pix.shape[0], ny)
        self.assertEqual(img2.pix.shape[1], nx)
        self.assertTrue(img2.pix.shape == img2.ivar.shape)
        self.assertTrue(img2.pix.shape == img2.mask.shape)

        #- Slice and dice multiple ways, getting meta NAXIS1/NAXIS2 correct
        img1 = Image(self.pix, self.ivar, meta=meta)
        img2 = img1[0:ny]
        self.assertEqual(img2.pix.shape[0], ny)
        self.assertEqual(img2.pix.shape[1], img1.pix.shape[1])
        self.assertEqual(img2.pix.shape[0], img2.meta['NAXIS2'])
        self.assertEqual(img2.pix.shape[1], img2.meta['NAXIS1'])

        img2 = img1[0:ny, :]
        self.assertEqual(img2.pix.shape[0], ny)
        self.assertEqual(img2.pix.shape[1], img1.pix.shape[1])
        self.assertEqual(img2.pix.shape[0], img2.meta['NAXIS2'])
        self.assertEqual(img2.pix.shape[1], img2.meta['NAXIS1'])

        img2 = img1[:, 0:nx]
        self.assertEqual(img2.pix.shape[0], img1.pix.shape[0])
        self.assertEqual(img2.pix.shape[1], nx)
        self.assertEqual(img2.pix.shape[0], img2.meta['NAXIS2'])
        self.assertEqual(img2.pix.shape[1], img2.meta['NAXIS1'])

        #- test slicing readnoise image
        readnoise = np.random.uniform(size=self.pix.shape)
        img1 = Image(self.pix, self.ivar, meta=meta, readnoise=readnoise)
        xy = np.s_[1:1+ny, 2:2+nx]
        img2 = img1[xy]
        self.assertTrue(np.all(img1.pix[xy] == img2.pix))
        self.assertTrue(np.all(img1.ivar[xy] == img2.ivar))
        self.assertTrue(np.all(img1.mask[xy] == img2.mask))
        self.assertTrue(np.all(img1.readnoise[xy] == img2.readnoise))
        
        #- Test bad slicing
        with self.assertRaises(ValueError):
            img1['blat']
        with self.assertRaises(ValueError):
            img1[1:2, 3:4, 5:6]
        with self.assertRaises(ValueError):
            img1[1:2, 'blat']
        with self.assertRaises(ValueError):
            img1[None, 1:2]
