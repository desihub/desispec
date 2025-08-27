"""
test desispec.efftime
"""

import unittest
import numpy as np
from astropy.table import Table

from desispec.efftime import compute_efftime

class TestEffTime(unittest.TestCase):

    # setUpClass runs once at the start before any tests
    @classmethod
    def setUpClass(cls):
        #- nominal values from https://desi.lbl.gov/trac/wiki/SurveyOps/SurveySpeed
        t = Table()
        n = 1
        exptime = 1000
        t['EXPTIME'] = np.ones(n) * exptime
        sky_nom = 3.73 # nMgy/arcsec**2
        t['SKY_MAG_R_SPEC'] = 22.5 - 2.5*np.log10(sky_nom)  # 21.07072792047828
        t['EBV'] = 0.0
        t['TRANSPARENCY_GFA'] = 1.0
        t['AIRMASS'] = 1.0
        t['FIBERFAC_GFA'] = 1.0
        t['FIBERFAC_ELG_GFA'] = 1.0
        t['FIBERFAC_BGS_GFA'] = 1.0
        t['FIBER_FRACFLUX_GFA'] = 0.582
        t['FIBER_FRACFLUX_ELG_GFA'] = 0.424
        t['FIBER_FRACFLUX_BGS_GFA'] = 0.195

        cls.reference = t

    def test_efftime(self):
        t = self.reference.copy()
        exptime = t['EXPTIME'][0]

        #- reference values have some rounding, so only compare to 1e-4
        efftime_dark, efftime_bright, efftime_backup = compute_efftime(t)
        self.assertAlmostEqual(efftime_dark[0], exptime)
        self.assertAlmostEqual(efftime_bright[0], exptime)
        self.assertAlmostEqual(efftime_backup[0], exptime)

        #- half the transparency = half the signal but the same background
        #- efftime is 1/4 if S/N = S/sqrt(B)
        t['TRANSPARENCY_GFA'] = 0.5
        t['FIBERFAC_GFA'] = 0.5
        t['FIBERFAC_ELG_GFA'] = 0.5
        t['FIBERFAC_BGS_GFA'] = 0.5
        efftime_dark, efftime_bright, efftime_backup = compute_efftime(t)
        self.assertAlmostEqual(efftime_dark[0], exptime/4)
        self.assertAlmostEqual(efftime_bright[0], exptime/4)
        self.assertAlmostEqual(efftime_backup[0], exptime/4)


