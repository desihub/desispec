import unittest

import numpy as np
import desispec.io
from desispec.frame import Frame, Spectrum
from desispec.resolution import Resolution

class TestFrame(unittest.TestCase):

    def test_init(self):
        nspec = 3
        nwave = 10
        wave = np.arange(nwave)
        flux = np.random.uniform(size=(nspec, nwave))
        ivar = np.ones(flux.shape)
        mask = np.zeros(flux.shape, dtype=int)
        rdata = np.ones((nspec, 5, nwave))

        frame = Frame(wave, flux, ivar, mask, rdata, spectrograph=0)
        self.assertTrue(np.all(frame.wave == wave))
        self.assertTrue(np.all(frame.flux == flux))
        self.assertTrue(np.all(frame.ivar == ivar))
        self.assertTrue(np.all(frame.resolution_data == rdata))
        self.assertEqual(frame.nspec, nspec)
        self.assertEqual(frame.nwave, nwave)
        self.assertTrue(isinstance(frame.R[0], Resolution))
        #- check dimensionality mismatches
        self.assertRaises(AssertionError, lambda x: Frame(*x), (wave, wave, ivar, mask, rdata))
        self.assertRaises(AssertionError, lambda x: Frame(*x), (wave, flux[0:2], ivar, mask, rdata))
        
        #- Check constructing with defaults (must set fibers by some method)
        frame = Frame(wave, flux, ivar, spectrograph=0)
        self.assertEqual(frame.flux.shape, frame.mask.shape)
        
        #- Check usage of fibers inputs
        fibers = np.arange(nspec)
        frame = Frame(wave, flux, ivar, fibers=fibers)
        frame = Frame(wave, flux, ivar, fibers=fibers*2)
        manyfibers = np.arange(2*nspec)
        self.assertRaises(ValueError, lambda x: Frame(*x), (wave, flux, ivar, None, None, manyfibers))

        #- Check usage of meta
        meta = dict(BLAT=0, FOO='abc')
        frame = Frame(wave, flux, ivar, meta=meta, spectrograph=0)
        self.assertEqual(frame.meta['BLAT'], meta['BLAT'])
        self.assertEqual(frame.meta['FOO'], meta['FOO'])

        #- Check usage of spectrograph input
        for i in range(3):
            frame = Frame(wave, flux, ivar, spectrograph=i)
            self.assertEqual(len(frame.fibers), nspec)
            self.assertEqual(frame.fibers[0], i*500)

        # Check multi-mode assignment of fibers
        self.assertRaises(ValueError, lambda x: Frame(*x), (wave, flux, ivar, None, None, fibers, 1, meta))

        #- Check a fiber-assigning method is required
        self.assertRaises(ValueError, lambda x: Frame(*x), (wave, flux, ivar))

    def test_slice(self):
        nspec = 5
        nwave = 10
        wave = np.arange(nwave)
        flux = np.random.uniform(size=(nspec, nwave))
        ivar = np.ones(flux.shape)
        mask = np.zeros(flux.shape, dtype=int)
        rdata = np.ones((nspec, 5, nwave))
        fibermap = desispec.io.fibermap.empty_fibermap(nspec)

        frame = Frame(wave, flux, ivar, mask, rdata, spectrograph=0)
        x = frame[1]
        self.assertEqual(type(x), Spectrum)
        x = frame[1:2]
        self.assertEqual(type(x), Frame)
        x = frame[[1,2,3]]
        self.assertEqual(type(x), Frame)
        x = frame[frame.fibers<3]
        self.assertEqual(type(x), Frame)
        
        #- Slice fibermap too
        frame = Frame(wave, flux, ivar, mask, rdata, spectrograph=0, fibermap=fibermap)
        x = frame[frame.fibers<3]
        self.assertEqual(len(x.fibers), len(x.fibermap))
        x = frame[[1,2,3]]
        self.assertTrue(np.all(x.fibers == x.fibermap['FIBER']))

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
