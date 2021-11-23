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
            fm = assemble_fibermap(night, expid)['FIBERMAP'].data

            #- unmatched positioners aren't in coords files and have
            #- FIBER_X/Y == 0, but most should be non-zero
            self.assertLess(np.count_nonzero(fm['FIBER_X'] == 0.0), 50)
            self.assertLess(np.count_nonzero(fm['FIBER_Y'] == 0.0), 50)

            #- all with FIBER_X/Y == 0 should have a FIBERSTATUS flag
            ii = (fm['FIBER_X'] == 0.0) & (fm['FIBER_Y'] == 0.0)
            self.assertTrue(np.all(fm['FIBERSTATUS'][ii] != 0))

            #- and platemaker x/y shouldn't match fiberassign x/y
            self.assertTrue(np.all(fm['FIBER_X'] != fm['FIBERASSIGN_X']))
            self.assertTrue(np.all(fm['FIBER_Y'] != fm['FIBERASSIGN_Y']))

            #- spot check existence of a few other columns
            for col in (
                'TARGETID', 'LOCATION', 'FIBER', 'TARGET_RA', 'TARGET_DEC',
                'PLATE_RA', 'PLATE_DEC',
                ):
                self.assertIn(col, fm.columns.names)

    @unittest.skipUnless(standard_nersc_environment, "not at NERSC")
    def test_missing_input_files(self):
        """Test creation of fibermaps with missing input files"""
        #- missing coordinates file for this exposure
        night, expid = 20200219, 51053
        with self.assertRaises(FileNotFoundError):
            fm = assemble_fibermap(night, expid)

        #- But should work with force=True
        fm = assemble_fibermap(night, expid, force=True)

        #- ...albeit with FIBER_X/Y == 0
        assert np.all(fm['FIBERMAP'].data['FIBER_X'] == 0.0)
        assert np.all(fm['FIBERMAP'].data['FIBER_Y'] == 0.0)

    @unittest.skipUnless(standard_nersc_environment, "not at NERSC")
    def test_missing_input_columns(self):
        """Test creation of fibermaps with missing input columns"""
        #- second exposure of split, missing fiber location information
        #- in coordinates file, but info is present in previous exposure
        #- that was same tile and first in sequence
        fm1 = assemble_fibermap(20210406, 83714)['FIBERMAP'].data
        fm2 = assemble_fibermap(20210406, 83715)['FIBERMAP'].data

        def nanequal(a, b):
            """Compare two arrays treating NaN==NaN"""
            return np.equal(a, b, where=~np.isnan(a))

        assert np.all(nanequal(fm1['FIBER_X'], fm2['FIBER_X']))
        assert np.all(nanequal(fm1['FIBER_Y'], fm2['FIBER_Y']))
        assert np.all(nanequal(fm1['FIBER_RA'], fm2['FIBER_RA']))
        assert np.all(nanequal(fm1['FIBER_DEC'], fm2['FIBER_DEC']))
        assert np.all(nanequal(fm1['DELTA_X'], fm2['DELTA_X']))
        assert np.all(nanequal(fm1['DELTA_Y'], fm2['DELTA_Y']))

def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)

if __name__ == '__main__':
    unittest.main()
