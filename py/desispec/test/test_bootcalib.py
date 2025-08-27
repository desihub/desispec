"""
tests bootcalib code
"""

import unittest
from uuid import uuid1
import tempfile
import shutil
import os
import numpy as np
import glob
import locale
import requests
from astropy.io import fits

from desispec import bootcalib as desiboot
from desiutil import funcfits as dufits

from desispec.scripts import bootcalib as bootscript


class TestBoot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.origdir = os.getcwd()
        cls.testdir = tempfile.mkdtemp()
        os.chdir(cls.testdir)
        cls.testarc = 'test_arc.fits.gz'
        cls.testflat = 'test_flat.fits.gz'
        cls.testout = 'test_bootcalib_{}.fits'.format(uuid1())
        cls.qafile = 'test-qa-123jkkjiuc4h123h12h3423sadfew.pdf'
        cls.data_unavailable = False

        # Grab the data
        url_arc = 'https://data.desi.lbl.gov/public/epo/example_files/spectest/test_arc.fits.gz'
        url_flat = 'https://data.desi.lbl.gov/public/epo/example_files/spectest/test_flat.fits.gz'
        for url, outfile in [(url_arc, cls.testarc), (url_flat, cls.testflat)]:
            if not os.path.exists(outfile):
                try:
                    f = requests.get(url)
                except:
                    cls.data_unavailable = True
                else:
                    if f.status_code == 200:
                        with open(outfile, "wb") as code:
                            code.write(f.content)
                    else:
                        print('ERROR: Unable to fetch {}; HTTP status {}'.format(
                            url, f.status_code))
                        cls.data_unavailable = True

    @classmethod
    def tearDownClass(cls):
        #- Remove testdir only if it was created by tempfile.mkdtemp
        if cls.testdir.startswith(tempfile.gettempdir()) and os.path.exists(cls.testdir):
            shutil.rmtree(cls.testdir)

        os.chdir(cls.origdir)

    def setUp(self):
        os.chdir(self.testdir)

    def test_fiber_peaks(self):
        if self.data_unavailable:
            self.skipTest("Failed to download test data.")
        flat_hdu = fits.open(self.testflat)
        flat = flat_hdu[0].data
        ###########
        # Find fibers
        xpk, ypos, cut = desiboot.find_fiber_peaks(flat)
        assert len(xpk) == 25

    def test_tracing(self):
        if self.data_unavailable:
            self.skipTest("Failed to download test data.")
        flat_hdu = fits.open(self.testflat)
        flat = flat_hdu[0].data
        # Find fibers (necessary)
        xpk, ypos, cut = desiboot.find_fiber_peaks(flat)
        # Trace
        xset, xerr = desiboot.trace_crude_init(flat,xpk,ypos)
        xfit, fdicts = desiboot.fit_traces(xset,xerr)

    def test_gauss(self):
        if self.data_unavailable:
            self.skipTest("Failed to download test data.")
        flat_hdu = fits.open(self.testflat)
        flat = flat_hdu[0].data
        # Find fibers (necessary)
        xpk, ypos, cut = desiboot.find_fiber_peaks(flat)
        # Trace
        xset, xerr = desiboot.trace_crude_init(flat, xpk, ypos)
        xfit, fdicts = desiboot.fit_traces(xset, xerr)
        # Gaussian
        gauss = desiboot.fiber_gauss(flat, xfit, xerr)
        #import pdb
        #pdb.set_trace()
        np.testing.assert_allclose(np.median(gauss), 1.06, rtol=0.05)

    def test_parse_nist(self):
        """Test parsing of NIST arc line files.
        """
        tbl = desiboot.parse_nist('CdI')
        self.assertEqual(tbl['Ion'][0], 'CdI')

    def test_load_gdarc_lines(self):

        for camera in ['b', 'r', 'z']:
            llist = desiboot.load_arcline_list(camera)
            dlamb, gd_lines = desiboot.load_gdarc_lines(camera,llist)

    def test_wavelengths(self):
        if self.data_unavailable:
            self.skipTest("Failed to download test data.")
        # Read flat
        flat_hdu = fits.open(self.testflat)
        header = flat_hdu[0].header
        flat = flat_hdu[0].data
        ny = flat.shape[0]
        # Find fibers (necessary)
        xpk, ypos, cut = desiboot.find_fiber_peaks(flat)
        # Trace
        xset, xerr = desiboot.trace_crude_init(flat, xpk, ypos)
        xfit, fdicts = desiboot.fit_traces(xset, xerr)
        # Test fiber_gauss_old for coverage
        gauss = desiboot.fiber_gauss_old(flat, xfit, xerr)
        # Gaussian
        gauss = desiboot.fiber_gauss(flat, xfit, xerr)
        # Read arc
        arc_hdu = fits.open(self.testarc)
        arc = arc_hdu[0].data
        arc_ivar = np.ones(arc.shape)
        # Extract arc spectra (one per fiber)
        all_spec = desiboot.extract_sngfibers_gaussianpsf(arc, arc_ivar, xfit, gauss)
        # Line list
        camera = header['CAMERA']
        llist = desiboot.load_arcline_list(camera)
        dlamb, gd_lines = desiboot.load_gdarc_lines(camera,llist)
        #
        all_wv_soln = []
        for ii in range(1):
            spec = all_spec[:,ii]
            # Find Lines
            pixpk, flux = desiboot.find_arc_lines(spec)
            id_dict = {"pixpk":pixpk,"flux":flux}
            # Match a set of 5 gd_lines to detected lines
            desiboot.id_arc_lines_using_triplets(id_dict,gd_lines,dlamb)
            # Now the rest
            desiboot.id_remainder(id_dict, llist, deg=3)
            # Final fit wave vs. pix too
            final_fit, mask = dufits.iter_fit(np.array(id_dict['id_wave']),
                                              np.array(id_dict['id_pix']),
                                              'polynomial', 3, xmin=0., xmax=1.)
            rms = np.sqrt(np.mean((dufits.func_val(
                np.array(id_dict['id_wave'])[mask==0], final_fit)-
                                   np.array(id_dict['id_pix'])[mask==0])**2))
            final_fit_pix,mask2 = dufits.iter_fit(np.array(id_dict['id_pix']),
                                                  np.array(id_dict['id_wave']),
                                                  'legendre',4, niter=5)
            # Save
            id_dict['final_fit'] = final_fit
            id_dict['rms'] = rms
            id_dict['final_fit_pix'] = final_fit_pix
            id_dict['wave_min'] = dufits.func_val(0, final_fit_pix)
            id_dict['wave_max'] = dufits.func_val(ny-1, final_fit_pix)
            id_dict['mask'] = mask
            all_wv_soln.append(id_dict)

        self.assertLess(all_wv_soln[0]['rms'], 0.25)

    #- desispec.bootcalib.bootcalib may be redundant with
    #- desispec.scripts.bootcalib.main.  Include tests for both for now.

    #- bootcalib.bootcalib is broken; see https://github.com/desihub/desispec/issues/174
    @unittest.expectedFailure
    def test_bootcalib(self):
        from desispec.bootcalib import bootcalib
        from desispec.image import Image
        arc = fits.getdata(self.testarc)
        flat = fits.getdata(self.testflat)
        arcimage = Image(arc, np.ones_like(arc), camera='b0')
        flatimage = Image(flat, np.ones_like(flat), camera='b0')
        results = bootcalib(3, flatimage, arcimage)

    def test_main(self):
        if self.data_unavailable:
            self.skipTest("Failed to download test data.")
        argstr = [
            '--fiberflat', self.testflat,
            '--arcfile', self.testarc,
            '--outfile', self.testout,
            '--qafile', self.qafile,
        ]
        args = bootscript.parse(options=argstr)
        bootscript.main(args)

        #- Ensure the PSF class can read that file
        # from desispec.quicklook.qlpsf import PSF
        # psf = PSF(self.testout)

        #- Ensure the output format is useably by xytraceset
        from desispec.io import read_xytraceset
        psf = read_xytraceset(self.testout)

        #- While we're at it, test some PSF accessor functions
        indices = np.array([0,1])
        waves = np.array([psf.wavemin, psf.wavemin+1])
        allrows = np.arange(psf.npix_y)

        w = psf.wave_vs_y(fiber=0, y=indices)
        w = psf.wave_vs_y(fiber=1, y=allrows)

        x = psf.x_vs_wave(0, waves)
        y = psf.y_vs_wave(0, waves)


