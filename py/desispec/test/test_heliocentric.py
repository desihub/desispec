import unittest
import numpy as np
from astropy.table import Table
import astropy.constants
import astropy.units as u

from desispec.heliocentric import (barycentric_velocity_corr_kms,
                                   barycentric_velocity_multiplicative_corr,
                                   heliocentric_shift_res_data)

class TestHeliocentric(unittest.TestCase):

    def test_heliocentric_shift_res_data(self):
        """Test the resolution matrix shift functionality"""
        
        # Setup mock data
        nwave = 100
        ndiag = 11
        wave = np.linspace(4000, 5000, nwave)
        
        nspec = 2
        res_data = np.zeros((nspec, ndiag, nwave))
        res_data[:, ndiag//2, :] = 1.0 # Delta function in the middle

        fmap = Table()
        fmap['TARGET_RA'] = np.array([0.0, 0.0])
        fmap['TARGET_DEC'] = np.array([0.0, 0.0])
        fmap['MJD'] = np.array([58800.0, 58800.0])
        
        # Case 1: zero shift
        mjd = 58800.0
        v_fiber = barycentric_velocity_corr_kms(0, 0, mjd)
        c_kms = astropy.constants.c.to(u.km/u.s).value
        # Use heliocor to exactly match the fiber velocity so vshift=0
        heliocor = 1.0 + v_fiber / c_kms
        
        # Fiber 1: heliocor matches fiber 0, but fiber 1 will have different RA/DEC/MJD 
        fmap['TARGET_RA'][1] = 10.0
        v_fiber1 = barycentric_velocity_corr_kms(10, 0, mjd)
        vshift1 = v_fiber1 - (heliocor - 1.0) * c_kms

        shifted_res = heliocentric_shift_res_data(fmap, res_data, wave, heliocor=heliocor)
        
        # Fiber 0 should have zero shift
        self.assertTrue(np.allclose(shifted_res[0, ndiag//2, :], 1.0))
        self.assertIn('HELIOCOR_OFFSET', fmap.colnames)
        self.assertAlmostEqual(fmap['HELIOCOR_OFFSET'][0], 0.0, places=6)

        # Fiber 1 should be shifted
        self.assertTrue(np.all(shifted_res[1, ndiag//2, :] < 1.0))
        self.assertAlmostEqual(fmap['HELIOCOR_OFFSET'][1], vshift1 / c_kms, places=6)

    def test_missing_cols(self):
        """Test robustness to missing columns"""
        nwave = 100
        ndiag = 11
        wave = np.linspace(4000, 5000, nwave)
        nspec = 2
        res_data = np.zeros((nspec, ndiag, nwave))
        res_data[:, ndiag//2, :] = 1.0

        # Empty fibermap
        fmap = Table()
        fmap['TARGET_RA'] = np.array([0.0, 0.0])
        
        shifted_res = heliocentric_shift_res_data(fmap, res_data, wave)
        
        # Should return copy of original
        self.assertTrue(np.all(shifted_res == res_data))
        self.assertIn('HELIOCOR_OFFSET', fmap.colnames)
        self.assertTrue(np.all(fmap['HELIOCOR_OFFSET'] == 0.0))

if __name__ == '__main__':
    unittest.main()
