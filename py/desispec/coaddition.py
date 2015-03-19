"""
Algorithms for co-addition of independent observations of the same object.
"""

from __future__ import division, absolute_import

import numpy as np
import scipy.sparse
import scipy.linalg

class Spectrum(object):
    """
    A reduced flux spectrum with an associated diagonal inverse covariance and resolution matrix.

    Objects of this type provide the inputs and outputs of co-addition.

    Args:
        wlen(numpy.ndarray): Array of shape (n,) wavelengths in Angstroms where the flux is tabulated.
        flux(numpy.ndarray): Array of shape (n,) flux densities in 1e-17 erg/s/cm**2 at each wavelength.
        ivar(numpy.ndarray): Array of shape (n,) inverse variances of flux at each wavelength.
        resolution(scipy.sparse.dia_matrix): Sparse array of shape(n,n) whose rows give the resolution at
            each wavelength. Uses the dia_matrix sparse format.
    """
    def __init__(self,wlen,flux,ivar,resolution):
        self.wlen = wlen
        self.flux = flux
        self.ivar = ivar
        self.resolution = resolution
        self._initialize()

    def _initialize(self):
        # Initialize the quantities we will accumulate during co-addition.
        icov = scipy.sparse.dia_matrix((self.ivar[np.newaxis,:],[0]),self.resolution.shape)
        self.Cinv = self.resolution.T.dot(icov.dot(self.resolution))
        self.Cinv_f = self.resolution.T.dot(self.ivar*self.flux)

    def _finalize(self):
        # Recalculate the deconvolved solution and resolution.
        self.ivar,R = decorrelate(self.Cinv.todense())
        R_it = scipy.linalg.inv(R.T)
        self.flux = R_it.dot(self.Cinv_f)/self.ivar
        self.resolution = scipy.sparse.dia_matrix(R)

    def __iadd__(self,other):
        """
        Coadd this spectrum with another spectrum of the same object that uses the same wavelength grid.

        Raises:
            RuntimeError: Cannot coadd different wavelength grids.
        """
        if not np.array_equal(self.wlen,other.wlen):
            raise RuntimeError('Cannot coadd different wavelength grids.')
        # Accumulate weighted deconvolved fluxes.
        self.Cinv = self.Cinv + other.Cinv # sparse matrices do not support +=
        self.Cinv_f += other.Cinv_f
        return self

def decorrelate(Cinv):
    """
    Decorrelate an inverse covariance using the matrix square root.

    Implements the decorrelation part of the spectroperfectionism algorithm described in
    Bolton & Schlegel 2009 (BS) http://arxiv.org/abs/0911.2689, w uses the matrix square root of
    Cinv to form a diagonal basis. This is generally a better choice than the eigenvector or
    Cholesky bases since it leads to more localized basis vectors, as described in
    Hamilton & Tegmark 2000 http://arxiv.org/abs/astro-ph/9905192.

    Args:
        Cinv(numpy.ndarray): Square array of inverse covariance matrix elements. The algorithm
            uses dense matrix operations so a sparse Cinv should be converted todense(). The matrix
            is assumed to be positive definite but we do not check this.

    Returns:
        tuple: Tuple ivar,R of uncorrelated flux inverse variances and the corresponding
            resolution matrix. These have shapes (nflux,) and (nflux,nflux) respectively.
            The rows of R give the resolution-convolved responses to unit flux for each
            wavelength bin. Note that R is returned in dense format but can be converted
            to an efficient sparse format using scipy.sparse.dia_matrix(R).
    """
    # Calculate the matrix square root of Cinv to diagonalize the flux errors.
    L,X = scipy.linalg.eigh((Cinv+Cinv.T)/2.)
    # Check that all eigenvalues are positive.
    assert np.all(L > 0), 'Found some negative Cinv eigenvalues.'
    # Check that the eigenvectors are orthonormal so that Xt.X = 1
    assert np.allclose(np.identity(len(L)),X.T.dot(X))
    Q = X.dot(np.diag(np.sqrt(L)).dot(X.T))
    # Check BS eqn.10
    assert np.allclose(Cinv,Q.dot(Q))
    # Calculate the corresponding resolution matrix and diagonal flux errors.
    s = np.sum(Q,axis=1)
    R = Q/s[:,np.newaxis]
    ivar = s**2
    # Check BS eqn.14
    assert np.allclose(Cinv,R.T.dot(np.diag(ivar).dot(R)))
    return ivar,R
