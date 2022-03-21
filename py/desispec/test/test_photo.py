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
  
        # sv3-dark (secondary), sv1-bright (primary), main-dark (secondary), main-dark (primary, south), main-dark (primary, north)
        input_cat = Table()
        input_cat['TARGETID'] = [929413348720641, 39627061832191195, 2253225356951552, 39627565329026280, 39633086782113662]
        input_cat['TARGET_RA'] = [216.615157366494, 59.237567466068484, 217.118753472706, 195.3373716438542, 230.3357972402121]
        input_cat['TARGET_DEC'] = [1.61111164514945, -31.46536118661697, 1.40073953879623, -9.135959353230087, 40.614361185571525]
        tileids = [280, 80638, 1086, 1573, 1766]
        self.input_cat = input_cat
        self.tileids

    @unittest.skipUnless(standard_nersc_environment, "not at NERSC")
    def test_gather_targetdirs(self):
        """Test that we get the correct targeting directories given a tile."""
        truedirs = {
            # sv1
            '80613': ['/global/cfs/cdirs/desi/target/catalogs/dr9/0.47.0/targets/sv1/resolve/bright/'],
            # sv3 including ToOs
            '19': ['/global/cfs/cdirs/desi/survey/ops/surveyops/trunk/mtl/sv3/ToO/ToO.ecsv',
                   '/global/cfs/cdirs/desi/target/catalogs/dr9/0.57.0/targets/sv3/resolve/bright',
                   '/global/cfs/cdirs/desi/target/catalogs/dr9/0.57.0/targets/sv3/secondary/bright'],
            # main
            '2070': ['/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.1/targets/main/resolve/dark',
                     '/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.1/targets/main/secondary/dark']
                   }

        for tileid in truedirs.keys():
            targetdirs = gather_targetdirs(int(tileid))
            #print(tileid, targetdirs, truedirs[tileid])
            self.assertTrue(np.all(targetdirs == truedirs[tileid]))

    @unittest.skipUnless(standard_nersc_environment, "not at NERSC")
    def test_gather_targetphot(self):
        """Test that we get the correct targeting photometry for an input set of objects."""

        targetphot = gather_targetphot(input_cat, tileids=tileids)

        import pdb ; pdb.set_trace()

        truedirs = {
            # sv1
            '80613': ['/global/cfs/cdirs/desi/target/catalogs/dr9/0.47.0/targets/sv1/resolve/bright/'],
            # sv3 including ToOs
            '19': ['/global/cfs/cdirs/desi/survey/ops/surveyops/trunk/mtl/sv3/ToO/ToO.ecsv',
                   '/global/cfs/cdirs/desi/target/catalogs/dr9/0.57.0/targets/sv3/resolve/bright',
                   '/global/cfs/cdirs/desi/target/catalogs/dr9/0.57.0/targets/sv3/secondary/bright'],
            # main
            '2070': ['/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.1/targets/main/resolve/dark',
                     '/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.1/targets/main/secondary/dark']
                   }

        #import pdb ; pdb.set_trace()
        for tileid in truedirs.keys():
            targetdirs = gather_targetdirs(int(tileid))
            #print(tileid, targetdirs, truedirs[tileid])
            self.assertTrue(np.all(targetdirs == truedirs[tileid]))

    #@unittest.skipUnless(standard_nersc_environment, "not at NERSC")
    #def test_missing_input_files(self):
    #    """Test creation of fibermaps with missing input files"""
    #    #- missing coordinates file for this exposure
    #    night, expid = 20200219, 51053
    #    with self.assertRaises(FileNotFoundError):
    #        fm = assemble_fibermap(night, expid)
    #
    #    #- But should work with force=True
    #    fm = assemble_fibermap(night, expid, force=True)
    #
    #    #- ...albeit with FIBER_X/Y == 0
    #    assert np.all(fm['FIBERMAP'].data['FIBER_X'] == 0.0)
    #    assert np.all(fm['FIBERMAP'].data['FIBER_Y'] == 0.0)
    #
    #@unittest.skipUnless(standard_nersc_environment, "not at NERSC")
    #def test_missing_input_columns(self):
    #    """Test creation of fibermaps with missing input columns"""
    #    #- second exposure of split, missing fiber location information
    #    #- in coordinates file, but info is present in previous exposure
    #    #- that was same tile and first in sequence
    #    fm1 = assemble_fibermap(20210406, 83714)['FIBERMAP'].data
    #    fm2 = assemble_fibermap(20210406, 83715)['FIBERMAP'].data
    #
    #    def nanequal(a, b):
    #        """Compare two arrays treating NaN==NaN"""
    #        return np.equal(a, b, where=~np.isnan(a))
    #
    #    assert np.all(nanequal(fm1['FIBER_X'], fm2['FIBER_X']))
    #    assert np.all(nanequal(fm1['FIBER_Y'], fm2['FIBER_Y']))
    #    assert np.all(nanequal(fm1['FIBER_RA'], fm2['FIBER_RA']))
    #    assert np.all(nanequal(fm1['FIBER_DEC'], fm2['FIBER_DEC']))
    #    assert np.all(nanequal(fm1['DELTA_X'], fm2['DELTA_X']))
    #    assert np.all(nanequal(fm1['DELTA_Y'], fm2['DELTA_Y']))

def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)

if __name__ == '__main__':
    unittest.main()
