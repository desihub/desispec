"""
tests desispec.xytraceset
"""

import unittest
import importlib.resources

import numpy as np
from numpy.polynomial.legendre import legval

from desispec.xytraceset import XYTraceSet, get_badamp_fibers
from desispec.io import read_xytraceset

class TestSky(unittest.TestCase):

    def setUp(self):
        self.nspec = 10
        self.ncoef = 5
        self.wavemin = 5000
        self.wavemax = 5100
        self.npix_y = 200
        self.yy = np.arange(self.npix_y)

        self.waves = np.linspace(self.wavemin, self.wavemax)
        self.nwave = len(self.waves)

        # reduce waves -> (-1,1) range
        self.xw = 2*(self.waves-self.wavemin)/(self.wavemax-self.wavemin) - 1


    def tearDown(self):
        pass

    def test_minimal_xytraceset(self):
        xcoef = np.random.uniform(size=(self.nspec, self.ncoef))
        ycoef = np.random.uniform(size=(self.nspec, self.ncoef))

        xy = XYTraceSet(xcoef, ycoef, self.wavemin, self.wavemax, self.npix_y)

        x = xy.x_vs_wave(0, self.waves)
        self.assertEqual(len(x), self.nwave)
        self.assertTrue(np.allclose(x, legval(self.xw, xcoef[0])))

        y = xy.y_vs_wave(5, self.waves)
        self.assertEqual(len(y), self.nwave)
        self.assertTrue(np.allclose(y, legval(self.xw, ycoef[5])))

        w = xy.wave_vs_y(2, self.yy)
        self.assertEqual(len(w), self.npix_y)

        x = xy.x_vs_y(3, self.yy)
        self.assertEqual(len(x), self.npix_y)

        self.assertEqual(xy.npix_y, self.npix_y)
        self.assertEqual(xy.nspec, self.nspec)

        #- no xsigcoef/ysigcoef -> errors
        with self.assertRaises(RuntimeError):
            xsig = xy.xsig_vs_wave(0, self.waves)

        with self.assertRaises(RuntimeError):
            ysig = xy.ysig_vs_wave(0, self.waves)

    def test_xysig(self):
        xcoef = np.random.uniform(size=(self.nspec, self.ncoef))
        ycoef = np.random.uniform(size=(self.nspec, self.ncoef))
        xsigcoef = np.random.uniform(size=(self.nspec, self.ncoef))
        ysigcoef = np.random.uniform(size=(self.nspec, self.ncoef))
        meta = dict(blat=1, foo=2)

        xy = XYTraceSet(xcoef, ycoef, self.wavemin, self.wavemax, self.npix_y,
                        xsigcoef=xsigcoef, ysigcoef=ysigcoef, meta=meta)

        xsig = xy.xsig_vs_wave(0, self.waves)
        self.assertEqual(len(xsig), self.nwave)
        self.assertTrue(np.allclose(xsig, legval(self.xw, xsigcoef[0])))

        ysig = xy.ysig_vs_wave(5, self.waves)
        self.assertEqual(len(ysig), self.nwave)
        self.assertTrue(np.allclose(ysig, legval(self.xw, ysigcoef[5])))

        self.assertEqual(xy.meta['blat'], 1)
        self.assertEqual(xy.meta['foo'], 2)

    def test_xysig(self):
        xcoef = np.random.uniform(size=(self.nspec, self.ncoef))
        ycoef = np.random.uniform(size=(self.nspec, self.ncoef))
        xsigcoef = np.random.uniform(size=(self.nspec, self.ncoef))
        ysigcoef = np.random.uniform(size=(self.nspec, self.ncoef))
        meta = dict(blat=1, foo=2)

        #- first test without xsig, ysig
        xy = XYTraceSet(xcoef, ycoef, self.wavemin, self.wavemax, self.npix_y)

        xy2 = xy[1:3]

        # xy2 fiber 0 is original fiber 1
        x = xy2.x_vs_wave(0, self.waves)
        self.assertEqual(len(x), self.nwave)
        self.assertTrue(np.allclose(x, legval(self.xw, xcoef[1])))

        y = xy2.y_vs_wave(1, self.waves)
        self.assertEqual(len(y), self.nwave)
        self.assertTrue(np.allclose(y, legval(self.xw, ycoef[2])))

        #- now test with xsig, ysig
        xy = XYTraceSet(xcoef, ycoef, self.wavemin, self.wavemax, self.npix_y,
                        xsigcoef=xsigcoef, ysigcoef=ysigcoef)

        xy2 = xy[1:3]

        # xy2 fiber 0 is original fiber 1
        x = xy2.x_vs_wave(0, self.waves)
        self.assertEqual(len(x), self.nwave)
        self.assertTrue(np.allclose(x, legval(self.xw, xcoef[1])))

        y = xy2.y_vs_wave(1, self.waves)
        self.assertEqual(len(y), self.nwave)
        self.assertTrue(np.allclose(y, legval(self.xw, ycoef[2])))

        xsig = xy2.xsig_vs_wave(0, self.waves)
        self.assertEqual(len(xsig), self.nwave)
        self.assertTrue(np.allclose(xsig, legval(self.xw, xsigcoef[1])))

        ysig = xy2.ysig_vs_wave(1, self.waves)
        self.assertEqual(len(xsig), self.nwave)
        self.assertTrue(np.allclose(ysig, legval(self.xw, ysigcoef[2])))

        xsig = xy2.xsig_vs_y(0, self.yy)
        self.assertEqual(len(xsig), self.npix_y)

        ysig = xy2.ysig_vs_y(1, self.yy)
        self.assertEqual(len(xsig), self.npix_y)

    def test_badamp(self):
        psffile = importlib.resources.files('desispec').joinpath('test/data/ql/psf-r0.fits')
        xy = read_xytraceset(psffile)

        header = dict()
        fibers = get_badamp_fibers(header, xy)
        self.assertEqual(len(fibers), 0)

        header['CCDSECA'] = '[1:2057, 1:2064]'
        header['CCDSECB'] = '[2058:4114, 1:2064]'
        header['CCDSECC'] = '[1:2057, 2065:4128]'
        header['CCDSECD'] = '[2058:4114, 2065:4128]'

        header['BADAMPS'] = 'A'
        fibers = get_badamp_fibers(header, xy, verbose=True)
        self.assertEqual(len(fibers), 247)
        self.assertEqual(np.max(fibers), 246)

        header['BADAMPS'] = 'C'
        fibers = get_badamp_fibers(header, xy, verbose=False)
        self.assertEqual(len(fibers), 247)
        self.assertEqual(np.max(fibers), 246)
        
        header['BADAMPS'] = 'B'
        fibers = get_badamp_fibers(header, xy)
        self.assertEqual(len(fibers), 253)
        self.assertEqual(np.min(fibers), 247)

        header['BADAMPS'] = 'D'
        fibers = get_badamp_fibers(header, xy)
        self.assertEqual(len(fibers), 253)
        self.assertEqual(np.min(fibers), 247)

        header['BADAMPS'] = 'A,B'
        fibers = get_badamp_fibers(header, xy)
        self.assertEqual(len(fibers), 500)

        header['BADAMPS'] = 'CB'
        fibers = get_badamp_fibers(header, xy)
        self.assertEqual(len(fibers), 500)

