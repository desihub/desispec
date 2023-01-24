"""
desispec.skycorr
================

Class with sky model corrections
"""

class SkyCorr(object):
    def __init__(self, wave, dwave, dlsf, header):
        """Create SkyCorr object

        Args:
            wave  : 1D[nwave] wavelength in Angstroms
            dwave  : 2D[nspec, nwave] wavelength correction in Angstroms
            dlsf  : 2D[nspec, nwave] Line Spread Function correction in Angstroms
            header : (optional) header from FITS file HDU0
        All input arguments become attributes
        """
        assert wave.ndim == 1
        assert dwave.ndim == 2
        assert dwave.shape == dlsf.shape
        self.nspec, self.nwave = dwave.shape
        self.wave = wave
        self.dwave = dwave
        self.dlsf = dlsf
        self.header = header

class SkyCorrPCA(object):
    def __init__(self, dwave_mean, dwave_eigenvectors, dwave_eigenvalues,
                 dlsf_mean, dlsf_eigenvectors, dlsf_eigenvalues,
                 wave, header):

        """Create SkyCorrPCA object

        Args:
            dwave_mean  : 2D[nspec, nwave] wavelength mean correction in Angstroms
            dwave_eigenvectors  : 2D[ncomp,nspec, nwave] wavelength correction PCA comps in Angstroms
            dwave_eigenvalues  : 1D array (size can be larger than the number of saved eigenvectors)
            dlsf_mean  : 2D[nspec, nwave] LSF (line spread function) mean correction in Angstroms
            dlsf_eigenvectors  : 2D[ncomp,nspec, nwave] LSF meancorrection PCA comps  in Angstroms
            dlsf_eigenvalues  : 1D array (size can be larger than the number of saved eigenvectors)
            wave  : 1D[nwave] wavelength in Angstroms
            header : (optional) header from FITS file HDU0
        All input arguments become attributes
        """

        assert dwave_mean.ndim == 2
        assert dwave_eigenvectors.ndim == 3
        assert dwave_eigenvalues.ndim == 1
        assert dlsf_mean.ndim == 2
        assert dlsf_eigenvectors.ndim == 3
        assert dlsf_eigenvalues.ndim == 1
        assert wave.ndim == 1

        self.nspec, self.nwave = dwave_mean.shape
        self.wave       = wave
        self.dwave_mean = dwave_mean
        self.dwave_eigenvectors = dwave_eigenvectors
        self.dwave_eigenvalues  = dwave_eigenvalues

        self.dlsf_mean  = dlsf_mean
        self.dlsf_eigenvectors = dlsf_eigenvectors
        self.dlsf_eigenvalues  = dlsf_eigenvalues
        self.header     = header
