"""
tests desispec.io.photo

"""
import os
import unittest

import numpy as np
from desispec.io.photo import gather_targetphot, gather_tractorphot, gather_targetdirs
from desispec.io.meta import get_desi_root_readonly 

class TestPhoto(unittest.TestCase):

    def setUp(self):
        from astropy.table import Table

        # for testing gather_targetphot and gather_tractorphot
        
        # sv3-dark (secondary), sv1-bright (primary), main-dark (secondary), main-dark (primary, south), main-dark (primary, north), sv3-bright (TOO)
        input_cat = Table()
        input_cat['TARGETID'] = [929413348720641, 39627061832191195, 2253225356951552, 39627565329026280, 39633086782113662, 43977491483722156]
        input_cat['TARGET_RA'] = [216.615157366494, 59.237567466068484, 217.118753472706, 195.3373716438542, 230.3357972402121, 150.1306899]
        input_cat['TARGET_DEC'] = [1.61111164514945, -31.46536118661697, 1.40073953879623, -9.135959353230087, 40.614361185571525, 1.505752]
        input_cat['TILEID'] = [280, 80638, 1086, 1573, 1766, 19]
        self.input_cat = input_cat

        # targetphot results
        targetphot = Table() 
        targetphot['FLUX_R'] = np.array([0.0, 22.768417, 0.0, 2.0220177, 0.3754208, 0.0]).astype('f4')
        targetphot['FLUX_W1'] = np.array([0.0, 38.42719, 0.0, 8.39509, 1.7937177, 0.0]).astype('f4')
        targetphot['FLUX_IVAR_W2'] = np.array([0.0, 0.72876793, 0.0, 0.53684264, 1.5837417, 0.0]).astype('f4')
        targetphot['NUMOBS_INIT'] = np.array([1, 1, 1, 4, 2, 1]).astype(np.int64)
        self.targetphot = targetphot

        # tractorphot DR9 results
        tractorphot = Table() 
        tractorphot['FLUX_R'] = np.array([0.0, 22.768417, 0.4996462, 2.0220177, 0.3754208, 4.8209643]).astype('f4')
        tractorphot['FLUX_IVAR_W1'] = np.array([0.0, 2.6306653, 2.2135038, 2.3442872, 6.2124352, 2.5682714]).astype('f4')
        tractorphot['LS_ID'] = np.array([0, 9906610122001627, 9906622040377206, 9906617989139688, 9907735053993854, 9906622022814265]).astype(np.int64)
        self.tractorphot = tractorphot

        # tractorphot DR10 results
        input_cat = Table()
        input_cat['TARGETID'] = [39089837533316505, 39089837533316947, 39089837533316811]
        input_cat['TARGET_RA'] = [196.85695554, 196.89418677, 196.78531078]
        input_cat['TARGET_DEC'] = [-25.37735882, -25.26812684, -25.2172812]
        input_cat['TILEID'] = [83390, 83390, 83390] # NGC4993 tile with no DR9 photometry
        self.input_cat_dr10 = input_cat
        
        tractorphot = Table() 
        tractorphot['FLUX_G'] = np.array([3.1981304, 11.26542, 19.937193]).astype('f4')
        tractorphot['FLUX_I'] = np.array([9.856144, 56.05153 , 33.03303]).astype('f4')
        tractorphot['FLUX_IVAR_W3'] = np.array([0.0014487484, 0.0010963874, 0.001165816]).astype('f4')
        tractorphot['LS_ID'] = np.array([10995128657712508, 10995128743167117, 10995128743105186]).astype(np.int64)
        self.tractorphot_dr10 = tractorphot

    @unittest.skipUnless('NERSC_HOST' in os.environ, "not at NERSC")
    def test_gather_targetdirs(self):
        """Test that we get the correct targeting directories given a tile."""
        desi_root = get_desi_root_readonly()
        #surveyops_dir = os.environ['DESI_SURVEYOPS']
        surveyops_dir = desi_root+'/survey/ops/surveyops/trunk' # assumes a standard installation / environment
        truedirs = {
            # sv1
            '80613': np.array([desi_root+'/target/catalogs/dr9/0.47.0/targets/sv1/resolve/bright/']),
            # sv3 including ToOs
            '19': np.array([surveyops_dir+'/mtl/sv3/ToO/ToO.ecsv',
                            desi_root+'/target/catalogs/dr9/0.57.0/targets/sv3/resolve/bright',
                            desi_root+'/target/catalogs/dr9/0.57.0/targets/sv3/secondary/bright/sv3targets-bright-secondary.fits']),
            # main
            '2070': np.array([surveyops_dir+'/mtl/main/ToO/ToO.ecsv',
                              desi_root+'/target/catalogs/dr9/1.1.1/targets/main/resolve/dark',
                              desi_root+'/target/catalogs/dr9/1.1.1/targets/main/secondary/dark/targets-dark-secondary.fits']),                             
                   }

        for tileid in truedirs.keys():
            targetdirs = gather_targetdirs(int(tileid))
            print(tileid, targetdirs, truedirs[tileid])
            self.assertTrue(np.all(targetdirs == truedirs[tileid]))

    @unittest.skipUnless('NERSC_HOST' in os.environ, "not at NERSC")
    def test_gather_targetphot(self):
        """Test that we get the correct targeting photometry for an input set of objects."""

        targetphot = gather_targetphot(self.input_cat)
        for col in self.targetphot.colnames:
            self.assertTrue(np.all(targetphot[col] == self.targetphot[col]))

    @unittest.skipUnless('NERSC_HOST' in os.environ, "not at NERSC")
    def test_gather_tractorphot(self):
        """Test that we get the correct Tractor photometry for an input set of objects."""

        tractorphot = gather_tractorphot(self.input_cat)
        for col in self.tractorphot.colnames:
            self.assertTrue(np.all(tractorphot[col] == self.tractorphot[col]))

    @unittest.skipUnless('NERSC_HOST' in os.environ, "not at NERSC")
    def test_gather_tractorphot_dr10(self):
        """Like test_gather_tractorphot but for DR10 photometry."""
        desi_root = get_desi_root_readonly()
        legacysurveydir = os.path.join(desi_root, 'external', 'legacysurvey', 'dr10')
        tractorphot = gather_tractorphot(self.input_cat_dr10, legacysurveydir=legacysurveydir)
        for col in self.tractorphot_dr10.colnames:
            self.assertTrue(np.all(tractorphot[col] == self.tractorphot_dr10[col]))

if __name__ == '__main__':
    unittest.main()
