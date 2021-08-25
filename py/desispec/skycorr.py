"""
desispec.skycorr
============

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
