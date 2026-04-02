"""
tests desispec.io.fibermap.assemble_fibermap
"""

import os
import unittest
import tempfile

import numpy as np
from scipy.special import erf
from desispec.emlinefit import emlines_gaussfit, get_emlines

#- some tests require data only available at NERSC
_everest = '/global/cfs/cdirs/desi/spectro/redux/everest'
at_nersc = ('NERSC_HOST' in os.environ) and (os.path.exists(_everest))

class TestFibermap(unittest.TestCase):

    @unittest.skipUnless(at_nersc, "not at NERSC or everest prod missing") 
    def test_emlines_script(self):
        from desispec.scripts.emlinefit import main
        zdir = f'{_everest}/tiles/cumulative/1930/20210530'
        with tempfile.TemporaryDirectory() as outdir:
            outfile = f'{outdir}/emlines.fits'
            cmd = f'desi_emlinefit_afterburner --coadd {zdir}/coadd-0-1930-thru20210530.fits --redrock {zdir}/redrock-0-1930-thru20210530.fits --output {outfile} --bitnames ELG --emnames OII,OIII'
            options = cmd.split()[1:]
            main(options)
            self.assertTrue(os.path.exists(outfile))

    def test_get_emlines(self):
        """Basic test of get_emlines"""
        zspecs = [0.5, 1.0]
        waves = np.arange(5500, 7600)
        nspec = len(zspecs)
        nwave = len(waves)
        rand = np.random.RandomState(0)
        fluxes = rand.normal(size=(nspec, nwave))
        ivars = np.ones((nspec, nwave))

        results = get_emlines(zspecs, waves, fluxes, ivars)

        #- spot check basic dimensions
        for line in results.keys():
            for key in ['FLUX', 'FLUX_IVAR', 'SIGMA', 'SIGMA_IVAR',
                        'SHARE', 'EW', 'CHI2']:
                self.assertEqual(len(results[line][key]), nspec,
                                 f'len({line}.{key}) != {nspec}')

        #- both redshifts should have an answer for OII
        self.assertFalse(np.isnan(results['OII']['FLUX'][0]))
        self.assertFalse(np.isnan(results['OII']['FLUX'][1]))

        #- but OIII should have NaN for second since it is off wavelength grid
        self.assertFalse(np.isnan(results['OIII']['FLUX'][0]))
        self.assertTrue(np.isnan(results['OIII']['FLUX'][1]))

    def _make_gauss_pixel(self, waves, sigma, F0, w0):
        """Helper: compute pixel-integrated Gaussian flux density using the same formula as gauss_nocont."""
        edges = np.zeros(len(waves) + 1)
        edges[1:-1] = 0.5 * (waves[:-1] + waves[1:])
        edges[0] = waves[0] - 0.5 * (waves[1] - waves[0])
        edges[-1] = waves[-1] + 0.5 * (waves[-1] - waves[-2])
        sqrt2s = np.sqrt(2.0) * sigma
        pixel_widths = edges[1:] - edges[:-1]
        integrated = F0 * (erf((edges[1:] - w0) / sqrt2s) - erf((edges[:-1] - w0) / sqrt2s)) / 2.0
        return integrated / pixel_widths

    def test_flux_normalization(self):
        """Test that emlines_gaussfit recovers correct integrated flux and CHI2 for a known Gaussian."""
        rng = np.random.RandomState(0)
        # Use a coarse 2 A grid around HALPHA at z=0
        waves = np.arange(6350.5, 6780.5, 2.0)
        w0 = 6564.613  # HALPHA rest-frame wavelength
        true_sigma = 3.5
        true_flux = 10.0
        true_cont = 1.0

        # Build pixel-integrated Gaussian model
        model = self._make_gauss_pixel(waves, true_sigma, true_flux, w0)

        # Verify the model integrates to true_flux over all pixels
        edges = np.zeros(len(waves) + 1)
        edges[1:-1] = 0.5 * (waves[:-1] + waves[1:])
        edges[0] = waves[0] - 0.5 * (waves[1] - waves[0])
        edges[-1] = waves[-1] + 0.5 * (waves[-1] - waves[-2])
        pixel_widths = edges[1:] - edges[:-1]
        self.assertAlmostEqual(np.sum(model * pixel_widths), true_flux, places=6)

        # Add noise so curve_fit produces a non-degenerate covariance;
        # noise_level=1e-4 gives SNR~1e5 so recovered parameters should be within 0.1% of true values
        noise_level = 1e-4
        fluxes = true_cont + model + rng.normal(0.0, noise_level, size=len(waves))
        ivars = np.ones(len(waves)) / noise_level ** 2

        emdict, succeed = emlines_gaussfit("HALPHA", 0.0, waves, fluxes, ivars)

        self.assertTrue(succeed)
        # Recovered integrated flux should match true_flux to within 0.1%
        self.assertAlmostEqual(emdict["FLUX"] / true_flux, 1.0, places=3)
        # Recovered sigma should match true_sigma to within 0.1%
        self.assertAlmostEqual(emdict["SIGMA"] / true_sigma, 1.0, places=3)

        # CHI2 must equal sum((model - data)^2 * ivar) / ndof; emdict['fluxes'] and
        # emdict['ivars'] are the fit-region subset of the input arrays, matching emdict['models']
        ndof = emdict["NDOF"]
        self.assertGreater(ndof, 0)
        expected_chi2 = np.sum((emdict["models"] - emdict["fluxes"]) ** 2 * emdict["ivars"]) / ndof
        self.assertAlmostEqual(emdict["CHI2"], expected_chi2, places=10)
        # For noise matching ivar the reduced chi2 has mean 1 and std sqrt(2/ndof) ~ 0.23 for ndof~38;
        # bounds [0.3, 3.0] allow for ~3-sigma variation
        self.assertGreater(emdict["CHI2"], 0.3)
        self.assertLess(emdict["CHI2"], 3.0)
