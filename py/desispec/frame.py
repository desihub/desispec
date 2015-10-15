"""
Lightweight wrapper class for spectra, to be returned by io.read_frame
"""

from __future__ import absolute_import, division

import numpy as np

from desispec.resolution import Resolution
from desispec.coaddition import Spectrum

# class Spectrum(object):
#     def __init__(self, wave, flux, ivar, mask=None, R=None):
#         """Lightweight wrapper of a single spectrum
#         
#         Args:
#             wave (1D ndarray): wavelength in Angstroms
#             flux (1D ndarray): flux (photons or ergs/s/cm^2/A)
#             ivar (1D ndarray): inverse variance of flux
#             R : Resolution object
#             
#         All args become attributes.  This is syntactic sugar.
#         """
#         self.wave = wave
#         self.flux = flux
#         self.ivar = ivar
#         if mask is None:
#             self.mask = np.zeros(self.flux.shape, dtype=int)
#         else:
#             self.mask = mask
#             
#         self.R = R
#         

class Frame(object):
    def __init__(self, wave, flux, ivar, mask=None, resolution_data=None,
                fibers=None, spectrograph=None, meta=None):
        """
        Lightweight wrapper for multiple spectra on a common wavelength grid

        x.wave, x.flux, x.ivar, x.mask, x.resolution_data, x.header, sp.R
        
        Args:
            wave: 1D[nwave] wavelength in Angstroms
            flux: 2D[nspec, nwave] flux
            ivar: 2D[nspec, nwave] inverse variance of flux
            mask: (optional) 2D[nspec, nwave] integer bitmask of flux.  0=good.
            resolution_data: (optional) 3D[nspec, ndiag, nwave]
                             diagonals of resolution matrix data
            fibers: (optional) ndarray of which fibers these spectra are
            spectrograph: (optional) integer, which spectrograph [0-9]        
            meta: (optional) dict-like object (e.g. FITS header from HDU0) 
                  Must include SPECMIN

        Notes:
            spectrograph input is used only if fibers is None.  In this case,
            it assumes nspec_per_spectrograph = flux.shape[0] and calculates
            the fibers array for this spectrograph, i.e.
            fibers = spectrograph * flux.shape[0] + np.arange(flux.shape[0])

        Attributes:
            All input args become object attributes.
            nspec : number of spectra, flux.shape[0]
            nwave : number of wavelengths, flux.shape[1]
            specmin : minimum fiber number
            R: array of sparse Resolution matrix objects converted
               from resolution_data
        """
        assert wave.ndim == 1
        assert flux.ndim == 2
        assert wave.shape[0] == flux.shape[1]
        assert ivar.shape == flux.shape
        assert (mask is None) or mask.shape == flux.shape
        assert (mask is None) or mask.dtype in \
            (int, np.int64, np.int32, np.uint64, np.uint32), "Bad mask type "+str(mask.dtype)

        self.wave = wave
        self.flux = flux
        self.ivar = ivar
        self.meta = meta
        self.nspec, self.nwave = self.flux.shape
        
        if mask is None:
            self.mask = np.zeros(flux.shape, dtype=np.uint32)
        else:
            self.mask = mask

        if resolution_data is not None:
            if resolution_data.ndim != 3 or \
               resolution_data.shape[0] != self.nspec or \
               resolution_data.shape[2] != self.nwave:
               raise ValueError("Wrong dimensions for resolution_data[nspec, ndiag, nwave]")

        #- Maybe setup non-None identity matrix resolution matrix instead?
        self.resolution_data = resolution_data
        if resolution_data is not None:
            self.R = np.array( [Resolution(r) for r in resolution_data] )

        self.spectrograph = spectrograph

        # Deal with Fibers (these must be set!)
        if fibers is not None:
            if len(fibers) != self.nspec:
                raise ValueError("len(fibers) != nspec ({} != {})".format(len(fibers), self.nspec))
            self.fibers = fibers
        else:
            self.fibers = None
        # If spectrograph given and fibers set, check these are consistent
        if self.spectrograph is not None: 
            if self.fibers is None:
                self.fibers = self.spectrograph*self.nspec + np.arange(self.nspec, dtype=int)
        # If meta given, check for specmin 
        if self.meta is not None:
            if 'SPECMIN' in self.meta.keys():
                if self.fibers is None:
                    self.fibers = self.meta['SPECMIN'] + np.arange(self.nspec, dtype=int)
                else:
                    assert np.min(self.fibers) >= self.meta['SPECMIN']

        # Require fibers is set!
        if self.fibers is None:
            raise ValueError("Must set fibers by one of the methods!")


         
    def __getitem__(self, index):
        """
        Return a subset of the spectra on this frame
        
        If index is an integer, return a single Spectrum object, otherwise
        return a Frame object with the subset of spectra that are sliced
        by index, which can be anything that can index or slice a numpy array.
        
        i.e.
            type(self[1:3]) == Frame
            type(self[1])   == Spectrum #- not Frame 
            
        This is analogous to how integers vs. slices or arrays return either
        scalars or arrays when indexing numpy.ndarray .
        """
        if isinstance(index, int):
            return Spectrum(self.wave, self.flux[index], self.ivar[index], self.mask[index], self.R[index])
        
        #- convert index to 1d array to maintain dimentionality of sliced arrays
        if not isinstance(index, slice):
            index = np.atleast_1d(index)

        if self.resolution_data is not None:
            rdata = self.resolution_data[index, :, :]
        else:
            rdata = None
        
        result = Frame(self.wave, self.flux[index], self.ivar[index],
                    self.mask[index], resolution_data=rdata,
                    fibers=self.fibers[index], spectrograph=self.spectrograph,
                    meta=self.meta)
        
        #- TODO:
        #- if we define fiber ranges in the fits headers, correct header
        
        return result
