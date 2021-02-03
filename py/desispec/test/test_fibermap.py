"""
tests desispec.io.fibermap.assemble_fibermap
"""

import os
import unittest

import numpy as np
from desispec.io.fibermap import assemble_fibermap

if 'NERSC_HOST' in os.environ and \
        os.getenv('DESI_SPECTRO_DATA') == '/global/cfs/cdirs/desi/spectro/data':
    standard_nersc_environment = True
else:
    standard_nersc_environment = False

class TestFibermap(unittest.TestCase):

    @unittest.skipUnless(standard_nersc_environment, "not at NERSC") 
    def test_assemble_fibermap(self):
        """Test creation of fibermaps from raw inputs"""
        for night, expid in [
            (20200219, 51039),  #- old SPS header
            (20200315, 55611),  #- new SPEC header
            ]:
            print(f'Creating fibermap for {night}/{expid}')
            fm = assemble_fibermap(night, expid)

            #- unmatched positioners aren't in coords files and have
            #- FIBER_X/Y == 0, but most should be non-zero
            self.assertLess(np.count_nonzero(fm['FIBER_X'] == 0.0), 5)
            self.assertLess(np.count_nonzero(fm['FIBER_Y'] == 0.0), 5)

            #- and platemaker x/y shouldn't match fiberassign x/y
            self.assertTrue(np.all(fm['FIBER_X'] != fm['FIBERASSIGN_X']))
            self.assertTrue(np.all(fm['FIBER_Y'] != fm['FIBERASSIGN_Y']))

            #- spot check existence of a few other columns
            for col in (
                'TARGETID', 'LOCATION', 'FIBER', 'TARGET_RA', 'TARGET_DEC',
                ):
                self.assertIn(col, fm.colnames)

    @unittest.skipUnless(standard_nersc_environment, "not at NERSC") 
    def test_missing_inputs(self):
        """Test creation of fibermaps with missing inputs"""
        #- missing coordinates file for this exposure
        night, expid = 20200219, 51053
        with self.assertRaises(FileNotFoundError):
            fm = assemble_fibermap(night, expid)

        #- But should work with force=True
        fm = assemble_fibermap(night, expid, force=True)

        #- ...albeit with FIBER_X/Y == 0
        assert np.all(fm['FIBER_X'] == 0.0)
        assert np.all(fm['FIBER_Y'] == 0.0)

def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)

if __name__ == '__main__':
    unittest.main()
