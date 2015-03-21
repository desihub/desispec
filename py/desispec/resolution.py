"""
Standardized handling of sparse wavelength resolution matrices.

Use `python -m desispec.resolution` to run unit tests.
"""

from __future__ import division, absolute_import

import numpy as np
import scipy.sparse

# The total number of diagonals that we keep in the sparse formats.
num_diagonals = 21

class Resolution(scipy.sparse.dia_matrix):
    """
    Canonical representation of a resolution matrix.

    Inherits all of the method of scipy.sparse.dia_matrix, including todense() for converting
    to a dense 2D numpy array of matrix elements, most of which will be zero (so you generally
    want to avoid this).

    Args:
        data: Must be in one of the following formats: (1) a scipy.sparse matrix in DIA format
            with the required diagonals (but not necessarily in the canoncial order); (2) a
            2D square numpy arrray (i.e., a dense matrix) whose non-zero values beyond
            num_diagonals will be silently dropped; or (3) a 2D numpy array whose first
            dimension is num_diagonals, which is assumed to encode the sparse diagonal values.
            The last format is the one used to store resolution matrices in FITS files.

    Raises:
        RuntimeError: Invalid input data for initializing a sparse resolution matrix.
    """
    def __init__(self,data):

        if scipy.sparse.isspmatrix_dia(data) and np.array_equal(np.sort(data.offsets)[::-1],self.offsets):
            # Input is already in DIA format with the required diagonals.
            # We just need to put the diagonals in the correct order.
            diag_order = np.argsort(data.offsets)[::-1]
            scipy.sparse.dia_matrix.__init__(self,(data.data[diag_order],self.offsets),data.shape)

        elif type(data) is np.ndarray and len(data.shape) == 2:
            n1,n2 = data.shape
            if n1 == num_diagonals and n2 > n1:
                scipy.sparse.dia_matrix.__init__(self,(data,self.offsets),(n2,n2))
            elif n1 == n2:
                sparse_data = np.zeros((num_diagonals,n1))
                for index,offset in enumerate(self.offsets):
                    where =  slice(offset,None) if offset >= 0 else slice(None,offset)
                    sparse_data[index,where] = np.diag(data,offset)
                scipy.sparse.dia_matrix.__init__(self,(sparse_data,self.offsets),(n1,n1))
            else:
                raise RuntimeError('Cannot initialize Resolution with array shape (%d,%d)' % (n1,n2))

        else:
            raise RuntimeError('Cannot initialize Resolution from %r' % data)

    def to_fits_array(self):
        """
        Convert to an array of sparse diagonal values.

        This is the format used to store resolution matrices in FITS files. Note that some
        values in the returned rectangular array do correspond to actual matrix elements
        since the diagonals get smaller as you move away from the central diagonal.
        As long as you treat this array as an opaque representation for FITS I/O, you
        don't care about this. To actually use the matrix, create a Resolution object
        from the fits array first.

        Returns:
            numpy.ndarray: An array of (num_diagonals,nbins) sparse matrix element values
                close to the diagonal.
        """
        return self.data

    """
    A list of off-diagonal offsets in the canonical order.
    """
    offsets = np.arange(num_diagonals//2,-(num_diagonals//2)-1,-1)

def run_unit_tests(n = 100):

    print 'Testing the Resolution class with n=%d...' % n

    assert len(Resolution.offsets) == num_diagonals,'Resolution.offsets has wrong size.'

    dense = np.arange(n*n).reshape(n,n)
    R1 = Resolution(dense)
    assert scipy.sparse.isspmatrix_dia(R1),'Resolution is not recognized as a scipy.sparse.dia_matrix.'

    R2 = Resolution(R1)
    assert np.array_equal(R1.toarray(),R2.toarray()),'Constructor broken for dia_matrix input.'

    R3 = Resolution(R1.data)
    assert np.array_equal(R1.toarray(),R3.toarray()),'Constructor broken for array data input.'

    sparse = scipy.sparse.dia_matrix((R1.data[::-1],R1.offsets[::-1]),(n,n))
    R4 = Resolution(sparse)
    assert np.array_equal(R1.toarray(),R4.toarray()),'Constructor broken for permuted offsets input.'

    R5 = Resolution(R1.to_fits_array())
    assert np.array_equal(R1.toarray(),R5.toarray()),'to_fits_array() is broken.'

if __name__ == '__main__':
    run_unit_tests()
