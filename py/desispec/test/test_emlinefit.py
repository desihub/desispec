"""
tests desispec.io.fibermap.assemble_fibermap
"""

import os
import unittest
import tempfile

import numpy as np
from desispec.emlinefit import get_emlines

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


if __name__ == '__main__':
    unittest.main()
