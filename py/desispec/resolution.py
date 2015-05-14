"""
desispec.resolution
===================

Standardized handling of sparse wavelength resolution matrices.

Use ``python -m desispec.resolution`` to run unit tests.
"""

from __future__ import division, absolute_import

import numpy as np
import scipy.sparse
from scipy.special import erf

# The total number of diagonals that we keep in the sparse formats when
# converting from a dense matrix
default_ndiag = 21

def _gauss_pix(x, mean=0.0, sigma=1.0):
    """
    Utility function to integrate Gaussian density within pixels
    
    Args:
        x (1D array): pixel centers
        mean (float): mean of Gaussian
        sigma (float): sigma of Gaussian
        
    Returns:
        array of integals of the Gaussian density in the pixels.
        
    Note:
        All pixels must be the same size
    """
    x = (np.asarray(x, dtype=float) - mean) / sigma
    dx = x[1]-x[0]
    if not np.allclose(np.diff(x), dx):
        raise ValueError('all pixels must have the same size')
        
    edges = np.concatenate([x-dx/2, x[-1:]+dx/2])
    assert len(edges) == len(x)+1
    
    y = erf(edges)
    return (y[1:] - y[0:-1])/2
    

class Resolution(scipy.sparse.dia_matrix):
    """Canonical representation of a resolution matrix.

    Inherits all of the method of scipy.sparse.dia_matrix, including todense()
    for converting to a dense 2D numpy array of matrix elements, most of which
    will be zero (so you generally want to avoid this).

    Args:
        data: Must be in one of the following formats listed below.

    Raises:
        ValueError: Invalid input for initializing a sparse resolution matrix.

    Data formats:

    1. a scipy.sparse matrix in DIA format with the required diagonals
       (but not necessarily in the canoncial order);
    2. a 2D square numpy arrray (i.e., a dense matrix) whose non-zero
       values beyond default_ndiag will be silently dropped; or
    3. a 2D numpy array[ndiag, nwave] that encodes the sparse diagonal
       values in the same format as scipy.sparse.dia_matrix.data .

    The last format is the one used to store resolution matrices in FITS files.

    """
    def __init__(self,data):

        ### if scipy.sparse.isspmatrix_dia(data) and np.array_equal(np.sort(data.offsets)[::-1],self.offsets):
        if scipy.sparse.isspmatrix_dia(data):
            # Input is already in DIA format with the required diagonals.
            # We just need to put the diagonals in the correct order.
            diag_order = np.argsort(data.offsets)[::-1]
            ndiag = len(data.offsets)
            if ndiag%2 == 0:
                raise ValueError("Number of diagonals ({}) should be odd".format(ndiag))
            self.offsets = np.arange(ndiag//2,-(ndiag//2)-1,-1)
            if not np.array_equal(data.offsets[diag_order], self.offsets):
                raise ValueError('Offsets of input matrix are non-contiguous or non-symmetric')
            scipy.sparse.dia_matrix.__init__(self,(data.data[diag_order],self.offsets),data.shape)

        elif isinstance(data,np.ndarray) and data.ndim == 2:
            n1,n2 = data.shape
            if n2 > n1:
                ndiag = n1  #- rename for clarity
                if ndiag%2 == 0:
                    raise ValueError("Number of diagonals ({}) should be odd".format(ndiag))
                self.offsets = np.arange(ndiag//2,-(ndiag//2)-1,-1)
                scipy.sparse.dia_matrix.__init__(self,(data,self.offsets),(n2,n2))
            elif n1 == n2:
                sparse_data = np.zeros((default_ndiag,n1))
                self.offsets = np.arange(default_ndiag//2,-(default_ndiag//2)-1,-1)
                for index,offset in enumerate(self.offsets):
                    where =  slice(offset,None) if offset >= 0 else slice(None,offset)
                    sparse_data[index,where] = np.diag(data,offset)
                scipy.sparse.dia_matrix.__init__(self,(sparse_data,self.offsets),(n1,n1))
            else:
                raise ValueError('Cannot initialize Resolution with array shape (%d,%d)' % (n1,n2))

        #- 1D data: Interpret as Gaussian sigmas in pixel units
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            nwave = len(data)
            rdata = np.empty((default_ndiag, nwave))
            self.offsets = np.arange(default_ndiag//2,-(default_ndiag//2)-1,-1)
            for i in range(nwave):
                rdata[:, i] = np.abs(_gauss_pix(self.offsets, mean=0.0, sigma=data[i]))
                
            scipy.sparse.dia_matrix.__init__(self,(rdata,self.offsets),(nwave,nwave))
            
        else:
            raise ValueError('Cannot initialize Resolution from %r' % data)

    def to_fits_array(self):
        """Convert to an array of sparse diagonal values.

        This is the format used to store resolution matrices in FITS files.
        Note that some values in the returned rectangular array do not
        correspond to actual matrix elements since the diagonals get smaller
        as you move away from the central diagonal. As long as you treat this
        array as an opaque representation for FITS I/O, you don't care about
        this. To actually use the matrix, create a Resolution object from the
        fits array first.

        Returns:
            numpy.ndarray: An array of (num_diagonals,nbins) sparse matrix
                element values close to the diagonal.
        """
        return self.data

#- (Unit tests moved to desispec.test.test_resolution)