"""
tests bootcalib code
"""

import unittest
from uuid import uuid1
import os

import numpy as np
import urllib2
import glob

from astropy.io import fits

from desispec import bootcalib as desiboot
from desiutil import funcfits as dufits

from desispec.scripts import bootcalib as bootscript


class TestBoot(unittest.TestCase):

    def setUp(self):
        self.testarc = 'test_arc.fits.gz'
        self.testflat = 'test_flat.fits.gz'
        self.testout = 'test_bootcalib_{}.fits'.format(uuid1())
        self.qafile = 'test-qa-123jkkjiuc4h123h12h3423sadfew.pdf'

        # Grab the data
        afil = glob.glob(self.testarc)
        if len(afil) == 0:
            url_arc = 'https://portal.nersc.gov/project/desi/data/spectest/pix-sub_b0-00000000.fits.gz'
            f = urllib2.urlopen(url_arc)
            with open(self.testarc, "wb") as code:
                code.write(f.read())
        ffil = glob.glob(self.testflat)
        if len(ffil) == 0:
            url_flat = 'https://portal.nersc.gov/project/desi/data/spectest/pix-sub_b0-00000001.fits.gz'
            f = urllib2.urlopen(url_flat)
            with open(self.testflat, "wb") as code:
                code.write(f.read())


    def tearDown(self):
        if os.path.isfile(self.testout):
            os.unlink(self.testout)

        if os.path.isfile(self.qafile):
            os.unlink(self.qafile)

    
    def test_fiber_peaks(self):
        flat_hdu = fits.open(self.testflat)
        flat = flat_hdu[0].data
        ###########
        # Find fibers
        xpk, ypos, cut = desiboot.find_fiber_peaks(flat)
        assert len(xpk) == 25

    def test_tracing(self):
        flat_hdu = fits.open(self.testflat)
        flat = flat_hdu[0].data
        # Find fibers (necessary)
        xpk, ypos, cut = desiboot.find_fiber_peaks(flat)
        # Trace
        xset, xerr = desiboot.trace_crude_init(flat,xpk,ypos)
        xfit, fdicts = desiboot.fit_traces(xset,xerr)

    def test_gauss(self):
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

    def test_load_gdarc_lines(self):
        
        for camera in ['b', 'r', 'z']:
            llist = desiboot.load_arcline_list(camera)
            dlamb, gd_lines = desiboot.load_gdarc_lines(camera,llist)

    def test_wavelengths(self):
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
        argstr = [
            '--fiberflat', self.testflat,
            '--arcfile', self.testarc,
            '--outfile', self.testout,
            '--qafile', self.qafile,
        ]
        args = bootscript.parse(options=argstr)
        bootscript.main(args)
        
        #- Ensure the PSF class can read that file
        from desispec.psf import PSF
        psf = PSF(self.testout)
        
        #- While we're at it, test some PSF accessor functions
        w = psf.wavelength()
        w = psf.wavelength(ispec=0)
        w = psf.wavelength(ispec=[0,1])
        w = psf.wavelength(ispec=[0,1], y=0)
        w = psf.wavelength(ispec=[0,1], y=[0,1])
        
        x = psf.x()
        x = psf.x(ispec=0)
        x = psf.x(ispec=[0,1])
        x = psf.x(ispec=None, wavelength=psf.wmin)
        x = psf.x(ispec=1, wavelength=psf.wmin)
        x = psf.x(ispec=[0,1], wavelength=psf.wmin)
        x = psf.x(ispec=[0,1], wavelength=[psf.wmin, psf.wmin+1])

        y = psf.y(ispec=None, wavelength=psf.wmin)
        y = psf.y(ispec=0, wavelength=psf.wmin)
        y = psf.y(ispec=[0,1], wavelength=psf.wmin)
        y = psf.y(ispec=[0,1], wavelength=[psf.wmin, psf.wmin+1])
        
        t = psf.invert()        

    def runTest(self):
        pass
                
if __name__ == '__main__':
    unittest.main()
