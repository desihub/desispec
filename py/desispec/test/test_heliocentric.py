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
        mjd = 58800
        
        # Case 1: zero shift
        v_fiber = barycentric_velocity_corr_kms(0, 0, mjd)
        c_kms = astropy.constants.c.to(u.km/u.s).value
        # Use heliocor to exactly match the fiber velocity so vshift=0
        heliocor = 1.0 + v_fiber / c_kms
        
        # Fiber 1: heliocor matches fiber 0, but use different RA/DEC to get non-zero shift
        fmap['TARGET_RA'][1] = 10.0
        v_fiber1 = barycentric_velocity_corr_kms(10, 0, mjd)
        vshift1 = v_fiber1 - (heliocor - 1.0) * c_kms

        shifted_res, offset_array = heliocentric_shift_res_data(fmap, res_data, wave, heliocor=heliocor, mjd=mjd)
        
        # Fiber 0 should have zero shift
        self.assertTrue(np.allclose(shifted_res[0, ndiag//2, :], 1.0))
        self.assertAlmostEqual(offset_array[0], 0.0, places=6)

        # Fiber 1 should be shifted
        self.assertTrue(np.all(shifted_res[1, ndiag//2, :] < 1.0))
        self.assertAlmostEqual(offset_array[1], vshift1 / c_kms, places=6)

    def test_pixel_shift(self):
        """Test that the resolution matrix is shifted by exactly 1 pixel"""
        nwave = 50
        ndiag = 11
        mjd = 58800.0
        
        # Field center at (0,0), star at (1,0)
        ra0, dec0 = 0.0, 0.0
        ra1, dec1 = 1.0, 0.0
        
        v_field = barycentric_velocity_corr_kms(ra0, dec0, mjd)
        v_star = barycentric_velocity_corr_kms(ra1, dec1, mjd)
        v_diff = v_star - v_field # ~0.3114 km/s
        
        c_kms = astropy.constants.c.to(u.km/u.s).value
        wave_val = 5000.0
        
        # Set dwave exactly to the wavelength shift corresponding to v_diff
        dwave = np.abs(v_diff) / c_kms * wave_val
        wave = np.arange(wave_val - nwave//2 * dwave,
                         wave_val + nwave//2 * dwave, dwave)
        nwave = len(wave)
        
        res_data = np.zeros((1, ndiag, nwave))
        res_data[0, ndiag//2, :] = 1.0 # Delta function in the middle
        
        fmap = Table()
        fmap['TARGET_RA'] = [ra1]
        fmap['TARGET_DEC'] = [dec1]
        
        # heliocor represents the field correction applied during extraction
        heliocor = 1.0 + v_field / c_kms
        
        shifted_res, offset_array = heliocentric_shift_res_data(fmap, res_data, wave, heliocor=heliocor, mjd=mjd)
        
        # vshift = v_star - v_field = v_diff
        # dwave was set to |v_diff| / c * wave, so deltas = -v_diff / |v_diff| = -1.0
        # A shift of deltas = -1.0 pixels corresponds to moving the kernel peak
        # from index ndiag//2 to ndiag//2 - 1.
        
        mid = nwave // 2
        self.assertAlmostEqual(shifted_res[0, ndiag//2 - 1, mid], 1.0, places=5)
        self.assertAlmostEqual(shifted_res[0, ndiag//2, mid], 0.0, places=5)

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
        
        shifted_res, offset_array = heliocentric_shift_res_data(fmap, res_data, wave)
        
        # Should return copy of original
        self.assertTrue(np.all(shifted_res == res_data))
        self.assertTrue(np.all(offset_array == 0.0))

if __name__ == '__main__':
    unittest.main()
