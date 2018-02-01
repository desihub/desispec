"""
desispec.resolution
===================

Standardized handling of sparse wavelength resolution matrices.

Use ``python -m desispec.resolution`` to run unit tests.
"""

from __future__ import division, absolute_import

import numpy as np
import scipy.sparse
import scipy.special

# The total number of diagonals that we keep in the sparse formats when
# converting from a dense matrix
default_ndiag = 21

def _sort_and_symmeterize(data, offsets):
    '''
    Sort data,offsets and pad to ensure equal number of upper/lower diagonals

    Args:
        data : 2D array of diagonals, following scipy.sparse.dia_matrix.data ordering
        offsets : 1D array of offsets; must be complete from min to max but
            doesn't have to be sorted

    Returns:
        fulldata, fulloffsets
    '''
    offsets = np.asarray(offsets)
    #- Already good?
    if np.all(np.diff(offsets) == -1) and offsets[0] == -offsets[-1]:
        return data, offsets

    #- Sort offsets and check for missing ones
    diag_order = np.argsort(offsets)[::-1]
    offsets = offsets[diag_order]
    if np.any(np.diff(offsets) != -1):
        raise ValueError('missing offsets {}'.format(offsets))

    #- Pad as needed to get equal number of upper and lower diagonals
    ndiag = np.max(np.abs(offsets))
    fulloffsets = np.arange(ndiag, -ndiag-1, -1)
    fulldata = np.zeros((2*ndiag+1, data.shape[1]), dtype=data.dtype)
    i = np.where(fulloffsets == offsets[0])[0][0]
    fulldata[i:i+len(offsets)] = data[diag_order]

    return fulldata, fulloffsets


class Resolution(scipy.sparse.dia_matrix):
    """Canonical representation of a resolution matrix.

    Inherits all of the method of scipy.sparse.dia_matrix, including todense()
    for converting to a dense 2D numpy array of matrix elements, most of which
    will be zero (so you generally want to avoid this).

    Args:
        data: Must be in one of the following formats listed below.

    Options:
        offsets: list of diagonals that the data represents.  Only used if
            data is a 2D dense array.

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
    def __init__(self, data, offsets=None):

        #- Sanity check on length of offsets
        if offsets is not None:
            if len(offsets) < 3:
                raise ValueError("Only {} resolution matrix diagonals?  That's probably way too small".format(len(offsets)))
            if len(offsets) > 4*default_ndiag:
                raise ValueError("{} resolution matrix diagonals?  That's probably way too big".format(len(offsets)))

        if scipy.sparse.isspmatrix_dia(data):
            # Input is already in DIA format with the required diagonals.
            # We just need to put the diagonals in the correct order.
            diadata, offsets = _sort_and_symmeterize(data.data, data.offsets)
            self.offsets = offsets
            scipy.sparse.dia_matrix.__init__(self, (diadata,offsets), data.shape)

        elif isinstance(data,np.ndarray) and data.ndim == 2:
            n1,n2 = data.shape
            if n2 > n1:
                ntotdiag = n1  #- rename for clarity
                if offsets is not None:
                    diadata, offsets = _sort_and_symmeterize(data, offsets)
                    self.offsets = offsets
                    scipy.sparse.dia_matrix.__init__(self, (diadata,offsets), (n2,n2))
                elif ntotdiag%2 == 0:
                    raise ValueError("Number of diagonals ({}) should be odd if offsets aren't included".format(ntotdiag))
                else:
                    #- Auto-derive offsets
                    self.offsets = np.arange(ntotdiag//2,-(ntotdiag//2)-1,-1)
                    scipy.sparse.dia_matrix.__init__(self,(data,self.offsets),(n2,n2))
            elif n1 == n2:
                if offsets is None:
                    self.offsets = np.arange(default_ndiag//2,-(default_ndiag//2)-1,-1)
                else:
                    self.offsets = np.sort(offsets)[-1::-1]  #- reverse sort

                sparse_data = np.zeros((len(self.offsets),n1))
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
                rdata[:, i] = np.abs(_gauss_pix(self.offsets, sigma=data[i]))

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
    x = (np.asarray(x, dtype=float) - mean) / (sigma*np.sqrt(2))
    dx = x[1]-x[0]
    if not np.allclose(np.diff(x), dx):
        raise ValueError('all pixels must have the same size')

    edges = np.concatenate([x-dx/2, x[-1:]+dx/2])
    assert len(edges) == len(x)+1

    y = scipy.special.erf(edges)
    return (y[1:] - y[:-1])/2



#- (Unit tests moved to desispec.test.test_resolution)
