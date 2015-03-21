"""
Algorithms for co-addition of independent observations of the same object.
"""

from __future__ import division, absolute_import

import numpy as np
import scipy.sparse
import scipy.linalg
import scipy.sparse.linalg

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
        # Initialize the quantities we will accumulate during co-addition.
        diag_ivar = scipy.sparse.dia_matrix((self.ivar[np.newaxis,:],[0]),self.resolution.shape)
        self.Cinv = self.resolution.T.dot(diag_ivar.dot(self.resolution))
        self.Cinv_f = self.resolution.T.dot(self.ivar*self.flux)

    def finalize(self,sparse_cutoff = 10):
        # Recalculate the deconvolved solution and resolution.
        self.ivar,R = decorrelate(self.Cinv)
        R_it = scipy.linalg.inv(R.T)
        self.flux = R_it.dot(self.Cinv_f)/self.ivar
        # Convert R from a dense matrix to a sparse one.
        n = len(self.ivar)
        k = int(sparse_cutoff)
        assert k >= 0,'Expected sparse_cutoff >= 0.'
        mask = np.tri(n,n,k) - np.tri(n,n,-k-1)
        self.resolution = scipy.sparse.dia_matrix(R*mask)

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

global_wavelength_grid = np.arange(3579.0,9825.0,1.0)

def combine(*spectra):
    """
    Combine a list of spectra using the global wavelength grid.
    """
    num_global = len(global_wavelength_grid)
    # Step 1: accumulated weighted sums of deconvolved flux estimates.
    Cinv = np.zeros((num_global,num_global))
    Cinv_f = np.zeros_like(global_wavelength_grid)
    Cinv_t = np.zeros_like(global_wavelength_grid)
    for spectrum in spectra:
        resampler = get_resampling_matrix(global_wavelength_grid,spectrum.wlen)
        Cinv += resampler.T.dot(spectrum.Cinv.dot(resampler))
        Cinv_f += resampler.T.dot(spectrum.Cinv_f)
    # Step 2: check for global wavelength bins with no information available.
    mask = (np.diag(Cinv) > 0)
    keep = np.arange(num_global)[mask]
    keep_t = keep[:,np.newaxis]
    # Step 3: find decorrelated basis.
    ivar = np.zeros_like(global_wavelength_grid)
    resolution = np.zeros_like(Cinv)
    ivar[mask],resolution[keep_t,keep] = decorrelate(Cinv[keep_t,keep])
    # Step 4: calculate decorrelated flux vectors.
    flux = np.zeros_like(global_wavelength_grid)
    R_it = scipy.linalg.inv(resolution[keep_t,keep].T)
    flux[mask] = R_it.dot(Cinv_f[mask])/ivar[mask]
    # Convert R from a dense matrix to a sparse one (ndiag = 21 hardcoded for now).
    ndiag = 21
    max_offset = ndiag//2
    offsets = np.arange(max_offset,-max_offset-1,-1)
    data = np.zeros((ndiag,num_global))
    for index,offset in enumerate(offsets):
        if offset >= 0:
            data[index,offset:] = np.diag(resolution,offset)
        else:
            data[index,:offset] = np.diag(resolution,offset)
    resolution = scipy.sparse.dia_matrix((data,offsets),resolution.shape)
    return flux,ivar,resolution

def get_resampling_matrix(global_grid,local_grid):
    """
    Build the rectangular matrix that linearly resamples from the global grid to a local grid.
    
    The local grid range must be contained within the global grid range.
    
    Args:
        global_grid(numpy.ndarray): Sorted array of n global grid wavelengths.
        local_grid(numpy.ndarray): Sorted array of m local grid wavelengths.

    Returns:
        numpy.ndarray: Array of (m,n) matrix elements that perform the linear resampling.
    """
    assert np.all(np.diff(global_grid) > 0),'Global grid is not strictly increasing.'
    assert np.all(np.diff(local_grid) > 0),'Local grid is not strictly increasing.'
    # Locate each local wavelength in the global grid.
    global_index = np.searchsorted(global_grid,local_grid)
    assert local_grid[0] >= global_grid[0],'Local grid extends below global grid.'
    assert local_grid[-1] <= global_grid[-1],'Local grid extends above global grid.'
    # Lookup the global-grid bracketing interval (xlo,xhi) for each local grid point.
    # Note that this gives xlo = global_grid[-1] if local_grid[0] == global_grid[0]
    # but this is fine since the coefficient of xlo will be zero.
    global_xhi = global_grid[global_index]
    global_xlo = global_grid[global_index-1]
    # Create the rectangular interpolation matrix to return.
    alpha = (local_grid - global_xlo)/(global_xhi - global_xlo)
    local_index = np.arange(len(local_grid),dtype=int)
    matrix = np.zeros((len(local_grid),len(global_grid)))
    matrix[local_index,global_index] = alpha
    matrix[local_index,global_index-1] = 1 - alpha
    return matrix

def decorrelate(Cinv):
    """
    Decorrelate an inverse covariance using the matrix square root.

    Implements the decorrelation part of the spectroperfectionism algorithm described in
    Bolton & Schlegel 2009 (BS) http://arxiv.org/abs/0911.2689, w uses the matrix square root of
    Cinv to form a diagonal basis. This is generally a better choice than the eigenvector or
    Cholesky bases since it leads to more localized basis vectors, as described in
    Hamilton & Tegmark 2000 http://arxiv.org/abs/astro-ph/9905192.

    Args:
        Cinv(numpy.ndarray): Square array of inverse covariance matrix elements. The input can
            either be a scipy.sparse format or else a regular (dense) numpy array, but a
            sparse format will be internally converted to a dense matrix so there is no
            performance advantage.

    Returns:
        tuple: Tuple ivar,R of uncorrelated flux inverse variances and the corresponding
            resolution matrix. These have shapes (nflux,) and (nflux,nflux) respectively.
            The rows of R give the resolution-convolved responses to unit flux for each
            wavelength bin. Note that R is returned as a regular (dense) numpy array but
            will normally have non-zero values concentrated near the diagonal.
    """
    # Clean up any roundoff errors by forcing Cinv to be symmetric.
    Cinv = 0.5*(Cinv + Cinv.T)
    # Convert to a dense matrix if necessary.
    if scipy.sparse.issparse(Cinv):
        Cinv = Cinv.todense()
    # Calculate the matrix square root. Note that we do not use scipy.linalg.sqrtm since
    # the method below is about 2x faster for a positive definite matrix.
    L,X = scipy.linalg.eigh(Cinv)
    # Check that all eigenvalues are positive.
    assert np.all(L > 0), 'Found some negative Cinv eigenvalues.'
    # Calculate the matrix square root Q such that Cinv = Q.Q
    Q = X.dot(np.diag(np.sqrt(L)).dot(X.T))
    # Calculate and return the corresponding resolution matrix and diagonal flux errors.
    s = np.sum(Q,axis=1)
    R = Q/s[:,np.newaxis]
    ivar = s**2
    return ivar,R
