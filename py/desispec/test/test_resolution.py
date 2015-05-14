"""
Unit tests for Resolution matrix creation
"""

import unittest
import numpy as np
import scipy.sparse

from desispec.resolution import Resolution
import desispec.resolution

class TestResolution(unittest.TestCase):

    #- TODO: in principle these should be converted to self.assertXYZ
    #- In practice, if they fail then the test won't pass so the end result is the same
    def test_resolution(self, n = 100):
        dense = np.arange(n*n).reshape(n,n)
        R1 = Resolution(dense)
        assert scipy.sparse.isspmatrix_dia(R1),'Resolution is not recognized as a scipy.sparse.dia_matrix.'
        assert len(R1.offsets) == desispec.resolution.default_ndiag, 'Resolution.offsets has wrong size'

        R2 = Resolution(R1)
        assert np.array_equal(R1.toarray(),R2.toarray()),'Constructor broken for dia_matrix input.'

        R3 = Resolution(R1.data)
        assert np.array_equal(R1.toarray(),R3.toarray()),'Constructor broken for array data input.'

        sparse = scipy.sparse.dia_matrix((R1.data[::-1],R1.offsets[::-1]),(n,n))
        R4 = Resolution(sparse)
        assert np.array_equal(R1.toarray(),R4.toarray()),'Constructor broken for permuted offsets input.'

        R5 = Resolution(R1.to_fits_array())
        assert np.array_equal(R1.toarray(),R5.toarray()),'to_fits_array() is broken.'

        #- test different sizes of input diagonals
        for ndiag in [3,5,11]:
            R6 = Resolution(np.ones((ndiag, n)))
            assert len(R6.offsets) == ndiag, 'Constructor broken for ndiag={}'.format(ndiag)

        #- An even number if diagonals is not allowed
        try:
            ndiag = 10
            R7 = Resolution(np.ones((ndiag, n)))
            raise RuntimeError('Incorrectly created Resolution with even number of diagonals')
        except ValueError, err:
            #- it correctly raised an error, so pass
            pass

        #- Test creation with asymetric diagonals (should fail)
        R1.offsets += 1
        try:
            R8 = Resolution(R1)
            raise RuntimeError('Incorrectly created Resolution with non-symmetric input')
        except ValueError:
            #- correctly raised an error, so pass
            pass
            
        #- Test creation with sigmas - it should conserve flux
        R9 = Resolution(np.linspace(1.0, 2.0, n))
        self.assertTrue(np.allclose(np.sum(R9.data, axis=0), 1.0))

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
