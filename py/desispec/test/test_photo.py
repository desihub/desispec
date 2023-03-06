"""
tests desispec.io.photo

"""
import os
import unittest

import numpy as np
from desispec.io.photo import gather_targetphot, gather_tractorphot, gather_targetdirs

if 'NERSC_HOST' in os.environ and \
        os.getenv('DESI_SPECTRO_DATA') == '/global/cfs/cdirs/desi/spectro/data':
    standard_nersc_environment = True
else:
    standard_nersc_environment = False

class TestFibermap(unittest.TestCase):

    def setUp(self):
        from astropy.table import Table

        # for testing gather_targetphot and gather_tractorphot
        
        # sv3-dark (secondary), sv1-bright (primary), main-dark (secondary), main-dark (primary, south), main-dark (primary, north)
        input_cat = Table()
        input_cat['TARGETID'] = [929413348720641, 39627061832191195, 2253225356951552, 39627565329026280, 39633086782113662]
        input_cat['TARGET_RA'] = [216.615157366494, 59.237567466068484, 217.118753472706, 195.3373716438542, 230.3357972402121]
        input_cat['TARGET_DEC'] = [1.61111164514945, -31.46536118661697, 1.40073953879623, -9.135959353230087, 40.614361185571525]
        self.input_cat = input_cat
        self.tileids = [280, 80638, 1086, 1573, 1766]

        # targetphot results
        targetphot = Table() 
        targetphot['FLUX_R'] = np.array([0.0, 22.768417, 0.0, 2.0220177, 0.3754208]).astype('f4')
        targetphot['FLUX_W1'] = np.array([0.0, 38.42719, 0.0, 8.39509, 1.7937177]).astype('f4')
        targetphot['FLUX_IVAR_W2'] = np.array([0.0, 0.72876793, 0.0, 0.53684264, 1.5837417]).astype('f4')
        targetphot['NUMOBS_INIT'] = np.array([1,1,1,4,9]).astype(np.int64)
        self.targetphot = targetphot

        # tractorphot results
        tractorphot = Table() 
        tractorphot['FLUX_R'] = np.array([0.0, 22.768417, 0.4996462, 2.0220177, 0.3754208]).astype('f4')
        tractorphot['FLUX_IVAR_W1'] = np.array([0.0, 2.6306653, 2.2135038, 2.3442872, 6.2124352]).astype('f4')
        tractorphot['LS_ID'] = np.array([0, 9906610122001627, 9906622040377206, 9906617989139688, 9907735053993854]).astype(np.int64)
        self.tractorphot = tractorphot

    @unittest.skipUnless(standard_nersc_environment, "not at NERSC")
    def test_gather_targetdirs(self):
        """Test that we get the correct targeting directories given a tile."""
        truedirs = {
            # sv1
            '80613': ['/global/cfs/cdirs/desi/target/catalogs/dr9/0.47.0/targets/sv1/resolve/bright/'],
            # sv3 including ToOs
            '19': ['/global/cfs/cdirs/desi/survey/ops/surveyops/trunk/mtl/sv3/ToO/ToO.ecsv',
                   '/global/cfs/cdirs/desi/target/catalogs/dr9/0.57.0/targets/sv3/resolve/bright',
                   '/global/cfs/cdirs/desi/target/catalogs/dr9/0.57.0/targets/sv3/secondary/bright/sv3targets-bright-secondary.fits'],
            # main
            '2070': ['/global/cfs/cdirs/desi/survey/ops/surveyops/trunk/mtl/main/ToO/ToO.ecsv',
                     '/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.1/targets/main/resolve/dark',
                     '/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.1/targets/main/secondary/dark/targets-dark-secondary.fits']
                   }

        for tileid in truedirs.keys():
            targetdirs = gather_targetdirs(int(tileid))
            #print(tileid, targetdirs, truedirs[tileid])
            self.assertTrue(np.all(targetdirs == truedirs[tileid]))

    @unittest.skipUnless(standard_nersc_environment, "not at NERSC")
    def test_gather_targetphot(self):
        """Test that we get the correct targeting photometry for an input set of objects."""

        targetphot = gather_targetphot(self.input_cat, tileids=self.tileids)
        for col in self.targetphot.colnames:
            self.assertTrue(np.all(targetphot[col] == self.targetphot[col]))

    @unittest.skipUnless(standard_nersc_environment, "not at NERSC")
    def test_gather_targetphot(self):
        """Test that we get the correct Tractor photometry for an input set of objects."""

        tractorphot = gather_tractorphot(self.input_cat)
        for col in self.tractorphot.colnames:
            self.assertTrue(np.all(tractorphot[col] == self.tractorphot[col]))

def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)

if __name__ == '__main__':
    unittest.main()
