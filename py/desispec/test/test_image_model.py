"""
test desispec.image_model
"""

import unittest
import importlib.resources

import numpy as np

from desispec.image_model import compute_image_model
from desispec.io import read_xytraceset
from desispec.image import Image
from desispec.fiberflat import FiberFlat

class TestImageModel(unittest.TestCase):

    # setUpClass runs once at the start before any tests
    @classmethod
    def setUpClass(cls):

        #- Read PSF and trim to just one bundle
        cls.camera = camera = 'r0'
        psffile = importlib.resources.files('desispec').joinpath(f'test/data/ql/psf-{camera}.fits')
        xy = read_xytraceset(psffile)
        nspec = 25
        xy = xy[0:nspec]

        wave = np.linspace(xy.wavemin, xy.wavemax)
        nwave = len(wave)
        xmin = int(np.min(xy.x_vs_wave(0, wave)))
        cls.nx = int(np.max(xy.x_vs_wave(xy.nspec-1, wave) + xmin))
        cls.ny = xy.npix_y
        cls.xy = xy

        ff = np.ones((nspec, nwave))
        ffivar = np.ones((nspec, nwave))
        cls.fiberflat = FiberFlat(wave, ff, ffivar)

    # tearDownClass runs once at the end after every test
    @classmethod
    def tearDownClass(cls):
        pass

    # setUp runs before every test
    def setUp(self):
        pass

    # setUp runs after every test
    def tearDown(self):
        pass

    def test_image_model(self):
        pix = np.random.normal(size=(self.ny, self.nx))
        ivar = np.ones((self.ny, self.nx))
        img = Image(pix, ivar, readnoise=1.0, camera=self.camera)

        model = compute_image_model(img, self.xy, fiberflat=self.fiberflat)


        


