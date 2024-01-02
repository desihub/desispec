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
        except ValueError as err:
            #- it correctly raised an error, so pass
            pass

        #- Test creation with sigmas - it should conserve flux
        R9 = Resolution(np.linspace(1.0, 2.0, n))
        self.assertTrue(np.allclose(np.sum(R9.data, axis=0), 1.0))

    def test_resolution_sparsedia(self):
        data = np.random.uniform(size=(5,10))
        offsets = np.arange(-2,3)

        #- Original case: symetric and odd number of diagonals
        Rdia = scipy.sparse.dia_matrix((data, offsets), shape=(10,10))
        R = Resolution(Rdia)
        self.assertTrue(np.all(R.diagonal() == Rdia.diagonal()))

        #- Non symetric but still odd number of diagonals
        Rdia = scipy.sparse.dia_matrix((data, offsets+1), shape=(10,10))
        R = Resolution(Rdia)
        self.assertTrue(np.all(R.diagonal() == Rdia.diagonal()))

        #- Even number of diagonals
        Rdia = scipy.sparse.dia_matrix((data[1:,:], offsets[1:]), shape=(10,10))
        R = Resolution(Rdia)
        self.assertTrue(np.all(R.diagonal() == Rdia.diagonal()))

        #- Unordered diagonals
        data = np.random.uniform(size=(5,10))
        offsets = [0,1,-1,2,-2]

        Rdia = scipy.sparse.dia_matrix((data, offsets), shape=(10,10))
        R1 = Resolution(Rdia)
        R2 = Resolution(data, offsets)
        self.assertTrue(np.all(R1.diagonal() == Rdia.diagonal()))
        self.assertTrue(np.all(R2.diagonal() == Rdia.diagonal()))
        self.assertTrue(np.all(R1.data == R2.data))

    def test_resolution_dense(self):
        #- dense with no offsets specified
        data = np.random.uniform(size=(10,10))
        R = Resolution(data)
        Rdense = R.todense()
        self.assertTrue(np.all(Rdense == data))        

        #- with offsets
        offsets = np.arange(-2,4)
        R = Resolution(data, offsets)
        Rdense = R.todense()
        for i in offsets:
            self.assertTrue(np.all(Rdense.diagonal(i) == data.diagonal(i)), \
                "diagonal {} doesn't match".format(i))

        #- dense without offsets but larger than default_ndiag
        ndiag = desispec.resolution.default_ndiag + 5
        data = np.random.uniform(size=(ndiag, ndiag))
        Rdense = Resolution(data).todense()

        for i in range(ndiag):
            if i <= desispec.resolution.default_ndiag//2:
                self.assertTrue(np.all(Rdense.diagonal(i) == data.diagonal(i)), \
                    "diagonal {} doesn't match".format(i))
                self.assertTrue(np.all(Rdense.diagonal(-i) == data.diagonal(-i)), \
                    "diagonal {} doesn't match".format(-i))
            else:
                self.assertTrue(np.all(Rdense.diagonal(i) == 0.0), \
                    "diagonal {} not 0s".format(i))
                self.assertTrue(np.all(Rdense.diagonal(-i) == 0.0), \
                    "diagonal {} not 0s".format(-i))

    def test_errors(self):
        #- Bad shaped input
        data = np.random.uniform(size=(10,5))
        with self.assertRaises(ValueError):
            R = Resolution(data)

        #- Meaningless type for input
        with self.assertRaises(ValueError):
            R = Resolution('blat')

        #- Non-uniform x spacing
        with self.assertRaises(ValueError):
            R = desispec.resolution._gauss_pix([-1,0,2])

        #- missing offsets
        with self.assertRaises(ValueError):
            desispec.resolution._sort_and_symmeterize(data, [-2,-1,0,1,3])

        #- length of offsets too large or small
        with self.assertRaises(ValueError):
            Resolution(data, offsets=[1,2])

        with self.assertRaises(ValueError):
            Resolution(data, offsets=np.arange(10*desispec.resolution.default_ndiag))

    def test_sort_and_symmeterize(self):
        #- if data,offsets are already good, just return them
        data = np.random.uniform((5,10))
        offsets = np.array([2,1,0,-1,-2])
        data2, offsets2 = desispec.resolution._sort_and_symmeterize(data, offsets)
        self.assertTrue(data is data2)
        self.assertTrue(offsets is offsets2)
