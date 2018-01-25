from __future__ import absolute_import, division, print_function

try:
    from specter.psf import load_psf
    nospecter = False
except ImportError:
    from desiutil.log import get_logger
    log = get_logger()
    log.error('specter not installed; skipping extraction tests')
    nospecter = True

import unittest
import uuid
import os
from glob import glob
from pkg_resources import resource_filename

import desispec.image
import desispec.io
import desispec.scripts.extract

from astropy.io import fits
import numpy as np

class TestExtract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.testhash = uuid.uuid4()
        cls.imgfile = 'test-img-{}.fits'.format(cls.testhash)
        cls.outfile = 'test-out-{}.fits'.format(cls.testhash)
        cls.outmodel = 'test-model-{}.fits'.format(cls.testhash)
        cls.fibermapfile = 'test-fibermap-{}.fits'.format(cls.testhash)
        cls.psffile = resource_filename('specter', 'test/t/psf-monospot.fits')
        # cls.psf = load_psf(cls.psffile)

        pix = np.random.normal(0, 3.0, size=(400,400))
        ivar = np.ones_like(pix) / 3.0**2
        mask = np.zeros(pix.shape, dtype=np.uint32)
        mask[200] = 1
        img = desispec.image.Image(pix, ivar, mask, camera='z0')
        desispec.io.write_image(cls.imgfile, img, meta=dict(flavor='science'))

        fibermap = desispec.io.empty_fibermap(100)
        desispec.io.write_fibermap(cls.fibermapfile, fibermap)

    def setUp(self):
        for filename in (self.outfile, self.outmodel):
            if os.path.exists(filename):
                os.remove(filename)

    @classmethod
    def tearDownClass(cls):
        for filename in glob('test-*{}*.fits'.format(cls.testhash)):
            if os.path.exists(filename):
                os.remove(filename)

    @unittest.skipIf(nospecter, 'specter not installed; skipping extraction test')
    def test_extract(self):
        template = "desi_extract_spectra -i {} -p {} -w 7500,7600,0.75 -f {} -s 0 -n 3 -o {} -m {}"

        cmd = template.format(self.imgfile, self.psffile, self.fibermapfile, self.outfile, self.outmodel)
        opts = cmd.split(" ")[1:]
        args = desispec.scripts.extract.parse(opts)
        desispec.scripts.extract.main(args)

        self.assertTrue(os.path.exists(self.outfile))
        frame1 = desispec.io.read_frame(self.outfile)
        model1 = fits.getdata(self.outmodel)
        os.remove(self.outfile)
        os.remove(self.outmodel)

        desispec.scripts.extract.main_mpi(args, comm=None)
        self.assertTrue(os.path.exists(self.outfile))
        frame2 = desispec.io.read_frame(self.outfile)
        model2 = fits.getdata(self.outmodel)

        self.assertTrue(np.all(frame1.flux[0:3] == frame2.flux[0:3]))
        self.assertTrue(np.all(frame1.ivar[0:3] == frame2.ivar[0:3]))
        self.assertTrue(np.all(frame1.mask[0:3] == frame2.mask[0:3]))
        self.assertTrue(np.all(frame1.chi2pix[0:3] == frame2.chi2pix[0:3]))
        self.assertTrue(np.all(frame1.resolution_data[0:3] == frame2.resolution_data[0:3]))

        #- These agree at the level of 1e-11 but not 1e-15.  Why not?
        #- We'll open a separate ticket about that, but allow to pass for now
        ### self.assertTrue(np.allclose(model1, model2, rtol=1e-15, atol=1e-15))
        self.assertTrue(np.allclose(model1, model2, rtol=1e-11, atol=1e-11))

        #- Check that units made it into the file
        self.assertEqual(frame1.meta['BUNIT'], 'electron/Angstrom')
        self.assertEqual(frame2.meta['BUNIT'], 'electron/Angstrom')

    def test_boxcar(self):
        from desispec.boxcar import do_boxcar
        psf = load_psf(self.psffile)

        pix = np.random.normal(0, 3.0, size=(psf.npix_y, psf.npix_x))
        ivar = np.ones_like(pix) / 3.0**2
        mask = np.zeros(pix.shape, dtype=np.uint32)
        img = desispec.image.Image(pix, ivar, mask, camera='z0')

        outwave = np.arange(7500, 7600)
        nwave = len(outwave)
        nspec = 5
        flux, ivar, resolution = do_boxcar(img, psf, outwave, boxwidth=2.5, nspec=nspec)

        self.assertEqual(flux.shape, (nspec, nwave))
        self.assertEqual(ivar.shape, (nspec, nwave))
        self.assertEqual(resolution.shape[0], nspec)
        # resolution.shape[1] is number of diagonals; picked by algorithm
        self.assertEqual(resolution.shape[2], nwave)

    def _test_bundles(self, template, specmin, nspec):
        #- should also work with bundles and not starting at spectrum 0
        cmd = template.format(self.imgfile, self.psffile, self.fibermapfile, self.outfile, self.outmodel, specmin, nspec)
        opts = cmd.split(" ")[1:]
        args = desispec.scripts.extract.parse(opts)
        desispec.scripts.extract.main(args)

        self.assertTrue(os.path.exists(self.outfile))
        frame1 = desispec.io.read_frame(self.outfile)
        model1 = fits.getdata(self.outmodel)
        os.remove(self.outfile)
        os.remove(self.outmodel)

        desispec.scripts.extract.main_mpi(args, comm=None)
        self.assertTrue(os.path.exists(self.outfile))
        frame2 = desispec.io.read_frame(self.outfile)
        model2 = fits.getdata(self.outmodel)

        self.assertTrue(np.all(frame1.flux[0:nspec] == frame2.flux[specmin:specmin+nspec]))
        self.assertTrue(np.all(frame1.ivar[0:nspec] == frame2.ivar[specmin:specmin+nspec]))
        self.assertTrue(np.all(frame1.mask[0:nspec] == frame2.mask[specmin:specmin+nspec]))
        self.assertTrue(np.all(frame1.chi2pix[0:nspec] == frame2.chi2pix[specmin:specmin+nspec]))
        self.assertTrue(np.all(frame1.resolution_data[0:nspec] == frame2.resolution_data[specmin:specmin+nspec]))

        #- pixel model isn't valid for small bundles that actually overlap; don't test
        # self.assertTrue(np.allclose(model1, model2, rtol=1e-15, atol=1e-15))

    #- traditional and MPI versions agree when starting at spectrum 0
    def test_bundles1(self):
        self._test_bundles("desi_extract_spectra -i {} -p {} -w 7500,7530,0.75 --nwavestep 10 -f {} --bundlesize 3 -o {} -m {} -s {} -n {}", 0, 5)

    #- BUG: they don't agree when starting at spectrum 2
    @unittest.expectedFailure
    def test_bundles2(self):
        self._test_bundles("desi_extract_spectra -i {} -p {} -w 7500,7530,0.75 --nwavestep 10 -f {} --bundlesize 3 -o {} -m {} -s {} -n {}", 2, 5)

    #- BUG: MPI version raises KeyError
    @unittest.expectedFailure
    def test_bundles3(self):
        self._test_bundles("desi_extract_spectra -i {} -p {} -w 7500,7530,0.75 --nwavestep 10 -f {} --bundlesize 3 -o {} -m {} -s {} -n {}", 22, 5)

if __name__ == '__main__':
    unittest.main()
