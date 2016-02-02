"""
tests bootcalib code
"""

import unittest

import numpy as np
import urllib2
import glob

from astropy.io import fits

from desispec import bootcalib as desiboot
from desiutil import funcfits as dufits

# Grab the data
arc_fil = 'tmp_arc.fits.gz'
afil = glob.glob(arc_fil)
if len(afil) == 0:
    url_arc = 'https://portal.nersc.gov/project/desi/data/spectest/pix-sub_b0-00000000.fits.gz'
    f = urllib2.urlopen(url_arc)
    with open(arc_fil, "wb") as code:
        code.write(f.read())
flat_fil = 'tmp_flat.fits.gz'
ffil = glob.glob(flat_fil)
if len(ffil) == 0:
    url_flat = 'https://portal.nersc.gov/project/desi/data/spectest/pix-sub_b0-00000001.fits.gz'
    f = urllib2.urlopen(url_flat)
    with open(flat_fil, "wb") as code:
        code.write(f.read())

class TestBoot(unittest.TestCase):
    
    def test_fiber_peaks(self):
        flat_hdu = fits.open(flat_fil)
        flat = flat_hdu[0].data
        ###########
        # Find fibers
        xpk, ypos, cut = desiboot.find_fiber_peaks(flat)
        assert len(xpk) == 25

    def test_tracing(self):
        flat_hdu = fits.open(flat_fil)
        flat = flat_hdu[0].data
        # Find fibers (necessary)
        xpk, ypos, cut = desiboot.find_fiber_peaks(flat)
        # Trace
        xset, xerr = desiboot.trace_crude_init(flat,xpk,ypos)
        xfit, fdicts = desiboot.fit_traces(xset,xerr)

    def test_gauss(self):
        flat_hdu = fits.open(flat_fil)
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

    def test_wavelengths(self):
        # Read flat
        flat_hdu = fits.open(flat_fil)
        header = flat_hdu[0].header
        flat = flat_hdu[0].data
        ny = flat.shape[0]
        # Find fibers (necessary)
        xpk, ypos, cut = desiboot.find_fiber_peaks(flat)
        # Trace
        xset, xerr = desiboot.trace_crude_init(flat, xpk, ypos)
        xfit, fdicts = desiboot.fit_traces(xset, xerr)
        # Gaussian
        gauss = desiboot.fiber_gauss(flat, xfit, xerr)
        # Read arc
        arc_hdu = fits.open(arc_fil)
        arc = arc_hdu[0].data
        # Extract arc spectra (one per fiber)
        all_spec = desiboot.extract_sngfibers_gaussianpsf(arc, xfit, gauss)
        # Line list
        camera = header['CAMERA']
        llist = desiboot.load_arcline_list(camera)
        dlamb, wmark, gd_lines, line_guess = desiboot.load_gdarc_lines(camera)
        #
        all_wv_soln = []
        for ii in range(1):
            spec = all_spec[:,ii]
            # Find Lines
            pixpk = desiboot.find_arc_lines(spec)
            # Match a set of 5 gd_lines to detected lines
            id_dict = desiboot.id_arc_lines(pixpk,gd_lines,dlamb,wmark,
                                            line_guess=line_guess)
            # Find the other good ones
            desiboot.add_gdarc_lines(id_dict, pixpk, gd_lines)
            # Now the rest
            desiboot.id_remainder(id_dict, pixpk, llist)
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

    def runTest(self):
        pass
                
if __name__ == '__main__':
    unittest.main()
