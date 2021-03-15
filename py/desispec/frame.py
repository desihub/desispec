"""
desispec.frame
==============

Lightweight wrapper class for spectra, to be returned by io.read_frame
"""

from __future__ import absolute_import, division

import numbers
import numpy as np

from desispec import util
from desispec.resolution import Resolution
from desiutil.log import get_logger
from desispec import util


class Spectrum(object):
    def __init__(self, wave, flux, ivar, mask=None, R=None):
        """Lightweight wrapper of a single spectrum
        
        Args:
             wave (1D ndarray): wavelength in Angstroms
             flux (1D ndarray): flux (photons or ergs/s/cm^2/A)
             ivar (1D ndarray): inverse variance of flux
             R : Resolution object

        All args become attributes.  This is syntactic sugar.
        """
        self.wave = wave
        self.flux = flux
        self.ivar = ivar
        if mask is None:
            self.mask = np.zeros(self.flux.shape, dtype=int)
        else:
            self.mask = mask
            self.R = R


class Frame(object):
    def __init__(self, wave, flux, ivar, mask=None, resolution_data=None,
                fibers=None, spectrograph=None, meta=None, fibermap=None,
                 chi2pix=None,scores=None,scores_comments=None,
                 wsigma=None,ndiag=21, suppress_res_warning=False
    ):
        """
        Lightweight wrapper for multiple spectra on a common wavelength grid

        x.wave, x.flux, x.ivar, x.mask, x.resolution_data, x.header, sp.R

        Args:
            wave: 1D[nwave] wavelength in Angstroms
            flux: 2D[nspec, nwave] flux
            ivar: 2D[nspec, nwave] inverse variance of flux

        Optional:
            mask: 2D[nspec, nwave] integer bitmask of flux.  0=good.
            resolution_data: 3D[nspec, ndiag, nwave]
                             diagonals of resolution matrix data
            fibers: ndarray of which fibers these spectra are
            spectrograph: integer, which spectrograph [0-9]
            meta: dict-like object (e.g. FITS header)
            fibermap: fibermap table
            chi2pix: 2D[nspec, nwave] chi2 of 2D model to pixel-level data
                for pixels that contributed to each flux bin
            scores: dictionary of 1D arrays of size nspec
            scores_comments: dictionnary of string (explaining the scores)
            suppress_res_warning: bool to suppress Warning message when the Resolution image is not read
        
        Parameters below allow on-the-fly resolution calculation
            wsigma: 2D[nspec,nwave] sigma widths for each wavelength bin for all fibers
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
            fibermap: fibermap table if provided
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
        self.fibermap = fibermap
        self.nspec, self.nwave = self.flux.shape
        self.chi2pix = chi2pix
        self.scores  = scores
        self.scores_comments  = scores_comments
        self.ndiag=ndiag
        fibers_per_spectrograph = 500   #- hardcode; could get from desimodel

        if mask is None:
            self.mask = np.zeros(flux.shape, dtype=np.uint32)
        else:
            self.mask = util.mask32(mask)

        if resolution_data is not None:
            if resolution_data.ndim != 3 or \
               resolution_data.shape[0] != self.nspec or \
               resolution_data.shape[2] != self.nwave:
               raise ValueError("Wrong dimensions for resolution_data[nspec, ndiag, nwave]")

        #- Maybe setup non-None identity matrix resolution matrix instead?
        self.wsigma=wsigma
        self.resolution_data = resolution_data
        if resolution_data is not None:
            self.wsigma=None #ignore width coefficients if resolution data is given explicitly
            self.ndiag=None 
            self.R = np.array( [Resolution(r) for r in resolution_data] )
        elif wsigma is not None:
            from desispec.quicklook.qlresolution import QuickResolution
            assert ndiag is not None
            r=[]
            for sigma in wsigma:
                r.append(QuickResolution(sigma=sigma,ndiag=self.ndiag))
            self.R=np.array(r)
        else:
            #SK I believe this should be error, but looking at the
            #tests frame objects are allowed to not to have resolution data
            # thus I changed value error to a simple warning message.
            if not suppress_res_warning:
                log = get_logger()
                log.warning("Frame object is constructed without resolution data or respective "\
                        "sigma widths. Resolution will not be available")
            # raise ValueError("Need either resolution_data or coefficients to generate it")
        self.spectrograph = spectrograph

        # Deal with Fibers (these must be set!)
        if fibers is not None:
            fibers = np.asarray(fibers)
            if len(fibers) != self.nspec:
                raise ValueError("len(fibers) != nspec ({} != {})".format(len(fibers), self.nspec))
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
                self.fibers = fibermap['FIBER']
            elif spectrograph is not None:
                self.fibers = spectrograph*fibers_per_spectrograph + np.arange(self.nspec, dtype=int)
            elif (self.meta is not None) and ('FIBERMIN' in self.meta):
                self.fibers = self.meta['FIBERMIN'] + np.arange(self.nspec, dtype=int)
            else:
                raise ValueError("Must set fibers by one of the methods!")

        if self.meta is not None:
            self.meta['FIBERMIN'] = np.min(self.fibers)

    def vet(self):
        """ Perform very basic checks on the frame
        Generally run before writing to disk (or when read)
        Args:
            index:

        Returns:
            diagnosis: int  (bitwise flag)
              0: Pass
              2**0: Improper meta data
              2**1: Improper data shapes

        """
        # Shapes
        log = get_logger()
        bad_shape = False
        if (self.nspec,self.nwave) != self.flux.shape:
            log.error('Frame nspec {} nwave {} inconsistent with flux.shape {}'.format(
                self.nspec, self.nwave, self.flux.shape))
            bad_shape = True

        # Meta data
        bad_meta = False
        if self.meta is None:
            log.error('Frame.meta missing')
            bad_meta = True
        else:
            from desispec.io.params import read_params
            # Check flavor
            if 'FLAVOR' not in self.meta.keys():
                log.error('Frame.meta missing FLAVOR keyword')
                bad_meta = True
            else:
                desi_params = read_params()
                if self.meta['FLAVOR'] not in desi_params['frame_types']:
                    log.error("Frame.meta['FLAVOR'] = '{}' not in {}".format(
                        self.meta['FLAVOR'], desi_params['frame_types']))
                    bad_meta = True
        #if bad_meta:
        #    import pdb; pdb.set_trace()

        # Generate the flag
        diagnosis = 0 + 2**0 * bad_shape + 2**1 * bad_meta
        return diagnosis

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
        if isinstance(index, numbers.Integral):
            return Spectrum(self.wave, self.flux[index], self.ivar[index], self.mask[index], self.R[index])

        #- convert index to 1d array to maintain dimentionality of sliced arrays
        if not isinstance(index, slice):
            index = np.atleast_1d(index)

        if self.resolution_data is not None:
            rdata = self.resolution_data[index, :, :]
        else:
            rdata = None

        if self.fibermap is not None:
            fibermap = self.fibermap[index]
        else:
            fibermap = None

        if self.chi2pix is not None:
            chi2pix = self.chi2pix[index]
        else:
            chi2pix = None

        #- we do not propagate the scores here
            
        wsigma=None
        if self.wsigma is not None:
            wsigma=self.wsigma[index]

        result = Frame(self.wave, self.flux[index], self.ivar[index],
                    self.mask[index], resolution_data=rdata,
                    fibers=self.fibers[index], spectrograph=self.spectrograph,
                       meta=self.meta, fibermap=fibermap, chi2pix=chi2pix,
                       wsigma=wsigma,ndiag=self.ndiag)

        return result

    def __repr__(self):
        txt = '<{:s}: nspec={:d}, nwave={:d}'.format(
            self.__class__.__name__, self.nspec, self.nwave)

        # Optional items
        if self.spectrograph is not None:
            txt + ', spectrograph={}'.format(self.spectrograph)
        if self.meta is not None:
            txt + ', FIBERMIN={}'.format(self.meta['FIBERMIN'])

        # Finish
        txt = txt + '>'
        return (txt)
