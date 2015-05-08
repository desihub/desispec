"""
Lightweight wrapper class for spectra, to be returned by io.read_frame
"""

from __future__ import absolute_import, division

import numpy as np

from desispec.resolution import Resolution

class Spectra(object):
    def __init__(self, wave, flux, ivar, resolution_data=None, header=None,
                fibers=None, spectrograph=0):
        """
        Creates a lightweight wrapper for spectra

        sp.wave, sp.flux, sp.ivar, sp.resolution_data, sp.header, sp.R
        
        Args:
            wave: 1D[nwave] wavelength in Angstroms
            flux: 2D[nspec, nwave] flux
            ivar: 2D[nspec, nwave] inverse variance of flux
            resolution_data: (optional) 3D[nspec, ndiag, nwave]
                             resolution matrix data
            header: (optional) FITS header from HDU0            
            
        Note:
            also converts resolution_data into R array of sparse Resolution
            matrix objects.
        """
        if wave.ndim != 1:
            raise ValueError("wave should be 1D")

        if flux.ndim != 2:
            raise ValueError("flux should be 2D[nspec, nwave]")

        if flux.shape != ivar.shape:
            raise ValueError("flux and ivar must have the same shape")
        
        if wave.shape[0] != flux.shape[1]:
            raise ValueError("nwave mismatch between wave.shape[0] and flux.shape[1]")

        self.wave = wave
        self.flux = flux
        self.ivar = ivar

        self.nspec, self.nwave = self.flux.shape

        if resolution_data is not None:
            if resolution_data.ndim != 3 or \
               resolution_data.shape[0] != self.nspec or \
               resolution_data.shape[2] != self.nwave:
               raise ValueError("Wrong dimensions for resolution_data[nspec, ndiag, nwave]")
        
        self.resolution_data = resolution_data
        if resolution_data is not None:
            self.R = np.array( [Resolution(r) for r in resolution_data] )

        self.header = header
        
        self.spectrograph = spectrograph
        if fibers is None:
            self.fibers = self.spectrograph + np.arange(self.nspec, dtype=int)
        else:
            if len(fibers) != self.nspec:
                raise ValueError("len(fibers) != nspec ({} != {})".format(len(fibers), self.nspec))
            self.fibers = fibers
            
    def __getitem__(self, index):
        """
        Return a subset of the spectra as a new Spectra object
        
        index can be anything that can index or slice a numpy array
        """
        #- convert index to 1d array to maintain dimentionality of sliced arrays
        if not isinstance(index, slice):
            index = np.atleast_1d(index)

        if self.resolution_data is not None:
            rdata = self.resolution_data[index, :, :]
        else:
            rdata = None
        
        result = Spectra(self.wave, self.flux[index], self.ivar[index],
                    resolution_data=rdata, header=self.header,
                    fibers=self.fibers[index], spectrograph=self.spectrograph)
        
        #- TODO:
        #- if we define fiber ranges in the fits headers, correct header
        
        return result
