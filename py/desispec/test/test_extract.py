from __future__ import absolute_import, division, print_function

try:
    from specter.psf import load_psf
    import gpu_specter
    nospecter = False
except ImportError:
    from desiutil.log import get_logger
    log = get_logger()
    log.error('specter and/or gpu_specter not installed; skipping extraction tests')
    nospecter = True

import unittest
import uuid
import os
import tempfile
from glob import glob
from importlib import resources

import desispec.image
import desispec.io
import desispec.scripts.extract

from astropy.io import fits
import numpy as np

class TestExtract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.origdir = os.getcwd()
        cls.testdir = tempfile.mkdtemp()
        os.chdir(cls.testdir)
        cls.testhash = uuid.uuid4()
        cls.imgfile = 'test-img-{}.fits'.format(cls.testhash)
        cls.outfile = 'test-out-{}.fits'.format(cls.testhash)
        cls.outmodel = 'test-model-{}.fits'.format(cls.testhash)
        cls.fibermapfile = 'test-fibermap-{}.fits'.format(cls.testhash)
        # cls.psffile = resources.files('specter').joinpath('test/t/psf-monospot.fits')
        cls.psffile = resources.files('gpu_specter').joinpath('test/data/psf-r0-00051060.fits')
        # cls.psf = load_psf(cls.psffile)

        pix = np.random.normal(0, 3.0, size=(4128, 4114))
        ivar = np.ones_like(pix) / 3.0**2
        mask = np.zeros(pix.shape, dtype=np.uint32)
        mask[200] = 1
        img = desispec.image.Image(pix, ivar, mask, camera='z0')
        desispec.io.write_image(cls.imgfile, img, meta=dict(flavor='science'))

        fibermap = desispec.io.empty_fibermap(100)
        desispec.io.write_fibermap(cls.fibermapfile, fibermap)

        cls.img = img

    def setUp(self):
        os.chdir(self.testdir)
        for filename in (self.outfile, self.outmodel):
            if os.path.exists(filename):
                os.remove(filename)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.testdir)
        for filename in glob('test-*{}*.fits'.format(cls.testhash)):
            if os.path.exists(filename):
                os.remove(filename)

        os.chdir(cls.origdir)

    @unittest.skipIf(nospecter, 'specter not installed; skipping extraction test')
    def test_extract(self):
        template = "desi_extract_spectra -i {} -p {} -w 7500,7600,0.75 -f {} -s 0 -n 5 --bundlesize 5 -o {} -m {}"

        cmd = template.format(self.imgfile, self.psffile, self.fibermapfile, self.outfile, self.outmodel)
        opts = cmd.split(" ")[1:]
        args = desispec.scripts.extract.parse(opts)
        desispec.scripts.extract.main(args)  #- gpu_specter

        self.assertTrue(os.path.exists(self.outfile))
        frame1 = desispec.io.read_frame(self.outfile)
        model1 = fits.getdata(self.outmodel)
        os.remove(self.outfile)
        os.remove(self.outmodel)

        desispec.scripts.extract.main_mpi(args, comm=None) #- specter
        self.assertTrue(os.path.exists(self.outfile))
        frame2 = desispec.io.read_frame(self.outfile)
        model2 = fits.getdata(self.outmodel)

        chi = (frame1.flux - frame2.flux) * np.sqrt(frame1.ivar)
        self.assertLess(np.max(np.abs(chi)), 0.05)

        #- specter and gpu_specter models of extracted pure noise don't agree
        #- TODO: add a test of modeling non-noise extration
        ### self.assertTrue(np.allclose(model1, model2, rtol=1e-15, atol=1e-15))
        ### self.assertTrue(np.allclose(model1, model2, rtol=1e-11, atol=1e-11))

        #- Check that units made it into the file
        self.assertEqual(frame1.meta['BUNIT'], 'electron/Angstrom')
        self.assertEqual(frame2.meta['BUNIT'], 'electron/Angstrom')

    def test_boxcar(self):
        from desispec.qproc.qextract import qproc_boxcar_extraction
        from desispec.io import read_xytraceset
        
        #psf = load_psf(self.psffile)
        tset = read_xytraceset(self.psffile)
        pix = np.random.normal(0, 3.0, size=(tset.npix_y, tset.npix_y))
        ivar = np.ones_like(pix) / 3.0**2
        mask = np.zeros(pix.shape, dtype=np.uint32)
        img = desispec.image.Image(pix, ivar, mask, camera='z0')

        outwave = np.arange(7500, 7600)
        nwave = len(outwave)
        nspec = 5
        fibers = np.arange(nspec)

        qframe = qproc_boxcar_extraction(tset, img, fibers=fibers)

        self.assertEqual(qframe.flux.shape, (nspec, tset.npix_y))
        self.assertEqual(qframe.ivar.shape, qframe.flux.shape)
        self.assertEqual(qframe.mask.shape, qframe.flux.shape)

    def _test_bundles(self, template, specmin, nspec):
        """
        Compare specter and gpu_specter extractions
        """
        #- should also work with bundles and not starting at spectrum 0
        cmd = template.format(self.imgfile, self.psffile, self.fibermapfile,
                self.outfile, self.outmodel, specmin, nspec)
        opts = cmd.split(" ")[1:]
        args = desispec.scripts.extract.parse(opts)
        desispec.scripts.extract.main(args)   #- defaults to gpu_specter

        self.assertTrue(os.path.exists(self.outfile))
        frame1 = desispec.io.read_frame(self.outfile)
        model1 = fits.getdata(self.outmodel)
        os.remove(self.outfile)
        os.remove(self.outmodel)

        opts = cmd.split(" ")[1:]
        opts.append('--use-specter')
        args = desispec.scripts.extract.parse(opts)
        desispec.scripts.extract.main(args)   #- specter

        self.assertTrue(os.path.exists(self.outfile))
        frame2 = desispec.io.read_frame(self.outfile)
        os.remove(self.outfile)
        os.remove(self.outmodel)

        desispec.scripts.extract.main_mpi(args, comm=None)  #- specter MPI path
        self.assertTrue(os.path.exists(self.outfile))
        frame3 = desispec.io.read_frame(self.outfile)
        model3 = fits.getdata(self.outmodel)

        errmsg = f'for specmin={specmin}, nspec={nspec}'

        #- specter and gpu_specter are consistent to 5% sigma
        chi = (frame1.flux - frame2.flux) * np.sqrt(frame1.ivar)
        self.assertLess(np.max(np.abs(chi)), 0.05)

        #- specter results should be the same
        self.assertTrue(np.allclose(frame2.flux, frame3.flux))
        self.assertTrue(np.allclose(frame2.ivar, frame3.ivar))
        self.assertTrue(np.allclose(frame2.chi2pix, frame3.chi2pix))
        self.assertTrue(np.allclose(frame2.resolution_data, frame3.resolution_data))
        self.assertTrue(np.all(frame2.mask == frame3.mask))

        #- pixel model isn't valid for small bundles that actually overlap; don't test
        # self.assertTrue(np.allclose(model1, model2, rtol=1e-15, atol=1e-15))

    #- traditional and MPI versions agree when starting at spectrum 0
    def test_bundles1(self):
        self._test_bundles("desi_extract_spectra -i {} -p {} -w 7500,7530,0.75 --nwavestep 10 -f {} --bundlesize 5 -o {} -m {} -s {} -n {}", 0, 5)

    #- test starting at a bundle non-boundary
    def test_bundles2(self):
        self._test_bundles("desi_extract_spectra -i {} -p {} -w 7500,7530,0.75 --nwavestep 10 -f {} --bundlesize 5 -o {} -m {} -s {} -n {}", 5, 5)

    def test_bundles3(self):
        self._test_bundles("desi_extract_spectra -i {} -p {} -w 7500,7530,0.75 --nwavestep 10 -f {} --bundlesize 5 -o {} -m {} -s {} -n {}", 20, 5)
