"""
desispec.qproc.qframe
==============

Lightweight wrapper class for spectra extract with qextract, row-by-row extraction (boxcar or profile)
"""

from __future__ import absolute_import, division

import numbers
import numpy as np

from desispec import util
from desispec.frame import Frame
from desispec.interpolation import resample_flux
from desiutil.log import get_logger

class QFrame(object):
    def __init__(self, wave, flux, ivar, mask=None, sigma=None,
                fibers=None, spectrograph=None, meta=None, fibermap=None,
    ):
        """
        Lightweight wrapper for multiple spectra 

        Args:
            wave: 2D[nspec, nwave] wavelength in Angstroms
            flux: 2D[nspec, nwave] flux
            ivar: 2D[nspec, nwave] inverse variance of flux

        Optional:
            mask: 2D[nspec, nwave] integer bitmask of flux.  0=good.
            sigma: 2D[nspec, nwave] LSF sigma in pixel units
            fibers: ndarray of which fibers these spectra are
            spectrograph: integer, which spectrograph [0-9]
            meta: dict-like object (e.g. FITS header)
            fibermap: fibermap table
            
        Attributes:
            All input args become object attributes.
            nspec : number of spectra, flux.shape[0]
            nwave : number of wavelengths, flux.shape[1]
            specmin : minimum fiber number
            R: array of sparse Resolution matrix objects converted
               from resolution_data
            fibermap: fibermap table if provided
        """
        assert wave.ndim == 2
        assert flux.ndim == 2
        assert wave.shape == flux.shape
        assert ivar.shape == flux.shape
        assert (mask is None) or mask.shape == flux.shape
        assert (mask is None) or mask.dtype in \
            (int, np.int64, np.int32, np.uint64, np.uint32), "Bad mask type "+str(mask.dtype)
        assert (sigma is None) or sigma.shape == flux.shape
        
        self.wave = wave
        self.flux = flux
        self.ivar = ivar
        self.meta = meta
        self.fibermap = fibermap
        if mask is None:
            self.mask = np.zeros(flux.shape, dtype=np.uint32)
        else:
            self.mask = util.mask32(mask)
        self.sigma = sigma
        
        self.nspec = self.flux.shape[0]
            
        self.spectrograph = spectrograph
        
        # Deal with Fibers (these must be set!)
        fibers_per_spectrograph = 500   #- hardcode; could get from desimodel
        if fibers is not None:
            fibers = np.asarray(fibers)
            if len(fibers) != self.flux.shape[0]:
                raise ValueError("len(fibers) != flux.shape[0] ({} != {})".format(len(fibers), flux.shape[0]))
            if fibermap is not None and np.any(fibers != fibermap['FIBER']):
                raise ValueError("fibermap doesn't match fibers")
            if (spectrograph is not None):
                minfiber = spectrograph*fibers_per_spectrograph
                maxfiber = (spectrograph+1)*fibers_per_spectrograph
                if np.any(fibers < minfiber) or np.any(maxfiber <= fibers):
                    raise ValueError('fibers inconsistent with spectrograph')
            self.fibers = fibers
        else:
            if fibermap is not None:
                self.fibers = np.asarray(fibermap['FIBER'])
            elif spectrograph is not None:
                self.fibers = spectrograph*fibers_per_spectrograph + np.arange(self.nspec, dtype=int)
            elif (self.meta is not None) and ('FIBERMIN' in self.meta):
                self.fibers = self.meta['FIBERMIN'] + np.arange(self.nspec, dtype=int)
            else:
                self.fibers = np.arange(self.flux.shape[0])

        if self.meta is not None:
            self.meta['FIBERMIN'] = np.min(self.fibers)

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
        
        #- convert index to 1d array to maintain dimentionality of sliced arrays
        if not isinstance(index, slice):
            index = np.atleast_1d(index)
        
        if self.fibermap is not None:
            fibermap = self.fibermap[index]
        else:
            fibermap = None
        
        if self.mask is None :
            tmp_mask=None
        else :
            tmp_mask=self.mask[index]
        
        if self.sigma is None :
            tmp_sigma=None
        else :
            tmp_sigma=self.sigma[index]
        
        result = QFrame(self.wave[index], self.flux[index], self.ivar[index],
                       tmp_mask, tmp_sigma, fibers=self.fibers[index], spectrograph=self.spectrograph,
                       meta=self.meta, fibermap=fibermap)
        
        return result


    def asframe(self,wavelength=None) :
        """
        Converts QFrame to a Frame 

        """

        if wavelength is None :
            dwave=np.min(np.gradient(self.wave[self.nspec//2]))
            wmin=np.max(self.wave[:,0])
            wmax=np.min(self.wave[:,-1])
            n=int((wmax-wmin)/dwave)+1
            wavelength=np.linspace(wmin,wmax,n)
        
        rflux = np.zeros((self.nspec,wavelength.size))
        rivar = np.zeros((self.nspec,wavelength.size))
        if self.mask is None :
            for i in range(self.nspec) :
                rflux[i],rivar[i] = resample_flux(wavelength,self.wave[i],self.flux[i],self.ivar[i],extrapolate=False)
        else :
            for i in range(self.nspec) :
                rflux[i],rivar[i] = resample_flux(wavelength,self.wave[i],self.flux[i],self.ivar[i]*(self.mask[i]==0),extrapolate=False)
        
            
        return Frame(wave=wavelength,flux=rflux,ivar=rivar,mask=None,resolution_data=None,\
                     fibers=self.fibers, spectrograph=None, meta=self.meta, fibermap=self.fibermap,\
                     chi2pix=None,scores=None,scores_comments=None,\
                     wsigma=self.sigma,ndiag=1, suppress_res_warning=True)
        
