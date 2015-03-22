"""
Algorithms for co-addition of independent observations of the same object.
"""

from __future__ import division, absolute_import

import numpy as np
import scipy.sparse
import scipy.linalg
import scipy.sparse.linalg

import desispec.resolution

class Spectrum(object):
    """
    A reduced flux spectrum with an associated diagonal inverse covariance and resolution matrix.

    Objects of this type provide the inputs and outputs of co-addition. When only a wavelength grid
    is passed to the constructor, the new object will represent a zero-flux spectrum. Use the +=
    operator to co-add spectra.

    Args:
        wlen(numpy.ndarray): Array of shape (n,) wavelengths in Angstroms where the flux is tabulated.
        flux(numpy.ndarray): Array of shape (n,) flux densities in 1e-17 erg/s/cm**2 at each wavelength.
        ivar(numpy.ndarray): Array of shape (n,) inverse variances of flux at each wavelength.
        resolution(desimodel.resolution.Resolution): Sparse matrix of wavelength resolutions.
    """
    def __init__(self,wlen,flux=None,ivar=None,resolution=None):
        self.wlen = wlen
        self.flux = flux
        self.ivar = ivar
        self.resolution = resolution
        # Initialize the quantities we will accumulate during co-addition. Note that our
        # internal Cinv is a dense matrix.
        if ivar is None:
            n = len(wlen)
            self.Cinv = np.zeros((n,n))
            self.Cinv_f = np.zeros((n,))
        else:
            assert flux is not None and resolution is not None,'Missing flux and/or resolution.'
            diag_ivar = scipy.sparse.dia_matrix((ivar[np.newaxis,:],[0]),resolution.shape)
            self.Cinv = self.resolution.T.dot(diag_ivar.dot(self.resolution))
            self.Cinv_f = self.resolution.T.dot(self.ivar*self.flux)

    def finalize(self):
        """
        Calculates the flux, inverse variance and resolution for this spectrum.

        Uses the accumulated data from all += operations so far but does not prevent
        further accumulation.  This is the expensive step in coaddition so we make
        it something that you have to call explicitly.  If you forget to do this,
        the flux,ivar,resolution attributes will be None.
        """
        # What pixels are we using?
        mask = (np.diag(self.Cinv) > 0)
        keep = np.arange(len(self.Cinv_f))[mask]
        keep_t = keep[:,np.newaxis]
        # Initialize the results to zero.
        self.flux = np.zeros_like(self.Cinv_f)
        self.ivar = np.zeros_like(self.Cinv_f)
        R = np.zeros_like(self.Cinv)
        # Calculate the deconvolved flux,ivar and resolution for ivar > 0 pixels.
        self.ivar[mask],R[keep_t,keep] = decorrelate(self.Cinv[keep_t,keep])
        R_it = scipy.linalg.inv(R[keep_t,keep].T)
        self.flux[mask] = R_it.dot(self.Cinv_f[mask])/self.ivar[mask]
        # Convert R from a dense matrix to a sparse one.
        self.resolution = desispec.resolution.Resolution(R)

    def __iadd__(self,other):
        """
        Coadd this spectrum with another spectrum.

        Raises:
            AssertionError: The other spectrum's wavelength grid is not compatible with ours.
        """
        # Accumulate weighted deconvolved fluxes.
        if np.array_equal(self.wlen,other.wlen):
            self.Cinv += other.Cinv
            self.Cinv_f += other.Cinv_f
        else:
            resampler = get_resampling_matrix(self.wlen,other.wlen)
            self.Cinv += resampler.T.dot(other.Cinv.dot(resampler))
            self.Cinv_f += resampler.T.dot(other.Cinv_f)
        # Make sure we don't forget to call finalize.
        self.flux = None
        self.ivar = None
        self.resolution = None

        return self

# The nominal brz grids cover 3579.0 - 9824.0 A but the FITs grids have some roundoff error
# so we add an extra bin to the end of the global wavelength grid to fully contain the bands.
global_wavelength_grid = np.arange(3579.0,9826.0,1.0)

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
