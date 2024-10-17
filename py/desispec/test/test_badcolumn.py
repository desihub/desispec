"""
Test desispec.badcolumn
"""

import unittest
import importlib.resources

import numpy as np
from astropy.table import Table

from desispec.frame import Frame
from desispec.io import read_xytraceset

class TestBadColumn(unittest.TestCase):

    # setUpClass runs once at the start before any tests
    @classmethod
    def setUpClass(cls):
        pass

    # setUpClass runs once at the end after every test
    # e.g. to remove files created by setUpClass
    @classmethod
    def setUpClass(cls):
        pass

    # setUp runs before every test, e.g. to reset state
    def setUp(self):
        pass

    # setUp runs after every test, e.g. to reset state
    def tearDown(self):
        pass

    def test_flux_bias_function(self):
        from desispec.badcolumn import flux_bias_function

        # scalar
        bias0 = flux_bias_function(0.0)
        bias1 = flux_bias_function(1)
        bias10 = flux_bias_function(10)
        self.assertTrue(np.isscalar(bias0))
        self.assertTrue(np.isscalar(bias1))
        self.assertLess(bias1, bias0)
        self.assertEqual(bias10, 0)
        
        # vector
        bias = flux_bias_function([0.0, 1, 10])
        self.assertEqual(bias[0], bias0)
        self.assertEqual(bias[1], bias1)
        self.assertEqual(bias[2], bias10)


    def test_compute_badcolumn_mask(self):
        from desispec.badcolumn import (
                compute_badcolumn_specmask, compute_badcolumn_fibermask, add_badcolumn_mask)

        #- Read a PSF and trim to just one bundle for faster testing
        psffile = importlib.resources.files('desispec').joinpath('test/data/ql/psf-r0.fits')
        xy = read_xytraceset(psffile)
        nspec = 25
        xy = xy[0:nspec]

        ny = xy.npix_y
        wave = np.arange(xy.wavemin, xy.wavemax)
        nwave = len(wave)
        minx = int(np.min(xy.x_vs_wave(0, wave)))
        nx = int(np.max(xy.x_vs_wave(xy.nspec-1, wave)) + minx)

        flux = np.zeros( (nspec, nwave) )
        ivar = np.ones( (nspec, nwave) )
        fibermap = Table()
        fibermap['FIBER'] = np.arange(nspec)
        fibermap['FIBERSTATUS'] = np.zeros(nspec, dtype=np.uint32)
        frame = Frame(wave, flux, ivar, fibermap=fibermap, meta=dict(CAMERA='r0'))

        badcol = nx//2
        badcol_table = Table()
        badcol_table['COLUMN'] = [badcol,]
        badcol_table['ELEC_PER_SEC'] = [1.,]

        specmask = compute_badcolumn_specmask(frame, xy, badcol_table)
        self.assertEqual(specmask.shape, (nspec,nwave))

        impacted_fibers = np.where(np.any(specmask, axis=1))[0]
        for i in impacted_fibers:
            #- flagged trace comes within 3 columns of the bad column
            dx = xy.x_vs_wave(i, wave) - badcol
            self.assertLess(np.min(np.abs(dx)), 3)

        #- masking at the fiber level requires a certain fraction of wavelenghts to be masked
        fibermask = compute_badcolumn_fibermask(specmask, camera_arm='r', threshold_specfrac=0.5)
        masked_fibers = np.where(fibermask != 0)[0]
        for i in range(nspec):
            frac_masked = np.sum(specmask[i]>0) / nwave
            if frac_masked >= 0.5:
                self.assertIn(i, masked_fibers)
            else:
                self.assertNotIn(i, masked_fibers)

        #- upper-case ok
        fibermask = compute_badcolumn_fibermask(specmask, camera_arm='R', threshold_specfrac=0.5)

        #- but camera must be b,r,z,B,R,Z
        with self.assertRaises(ValueError):
            fibermask = compute_badcolumn_fibermask(specmask, camera_arm='Q')

        #- Directly update frame mask
        self.assertTrue(np.all(frame.fibermap['FIBERSTATUS'] == 0))
        self.assertTrue(np.all(frame.mask == 0))
        add_badcolumn_mask(frame, xy, badcol_table)
        self.assertTrue(np.all(frame.mask == specmask))
        self.assertTrue(np.all(frame.fibermap['FIBERSTATUS'] == fibermask))

        #- Set a mask if it isn't already there
        frame.mask = None
        add_badcolumn_mask(frame, xy, badcol_table)
        self.assertTrue(np.all(frame.mask == specmask))

        #- len-0 badcol_table ok
        frame.mask *= 0
        frame.fibermap['FIBERSTATUS'] *= 0
        add_badcolumn_mask(frame, xy, badcol_table[0:0])
        self.assertTrue(np.all(frame.mask == 0))
        self.assertTrue(np.all(frame.fibermap['FIBERSTATUS'] == 0))


