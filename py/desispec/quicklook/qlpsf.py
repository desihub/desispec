"""
desispec.quicklook.qlpsf
========================

Given a psf output file e.g. output from bootcalib.write_psf or desimodel/data/specpsf/PSF files
this defines an interface that other codes can use the trace and wavelength solutions.

Mostly making parallel to specter.psf.PSF baseclass and inheriting as needed, but only xtrace,
ytrace and wavelength solution available for this case. No resolution information yet.
"""

import numbers
import numpy as np
from desiutil import funcfits as dufits
from numpy.polynomial.legendre import Legendre,legval,legfit
import astropy.io.fits as fits
import scipy.optimize

from desispec.io.xytraceset import read_xytraceset

class PSF(object):
    """
    Base class for 2D psf
    """
    def __init__(self,filename):


        print("desispec.psf is DEPRECATED, PLEASE USE desispec.xytraceset")

        self.traceset = read_xytraceset(filename)

        # all in traceset now.
        # psf kept to ease transition
        self.npix_y=self.traceset.npix_y
        self.xcoeff=self.traceset.x_vs_wave_traceset._coeff # in traceset
        self.ycoeff=self.traceset.y_vs_wave_traceset._coeff # in traceset
        self.wmin=self.traceset.wavemin # in traceset
        self.wmax=self.traceset.wavemax # in traceset
        self.nspec=self.traceset.nspec  # in traceset
        self.ncoeff=self.traceset.x_vs_wave_traceset._coeff.shape[1] #
        self.traceset.wave_vs_y(0,100.) # call wave_vs_y  for creation of wave_vs_y_traceset and consistent inversion
        self.icoeff=self.traceset.wave_vs_y_traceset._coeff  # in traceset
        self.ymin=self.traceset.wave_vs_y_traceset._xmin # in traceset
        self.ymax=self.traceset.wave_vs_y_traceset._xmax # in traceset


    def x(self,ispec=None,wavelength=None):
        """
        returns CCD x centroids for the spectra
        ispec can be None, scalar or a vector
        wavelength can be None or a vector
        """
        if ispec is None :
            ispec = np.arange(self.traceset.nspec)
        else :
            ispec = np.atleast_1d(ispec)

        if wavelength is None :
            wavelength = self.wavelength(ispec)
        else :
            wavelength = np.atleast_1d(wavelength)

        if len(wavelength.shape)==2 :
            res=np.zeros(wavelength.shape)
            for j,i in enumerate(ispec):
                res[j]=self.traceset.x_vs_wave(i,wavelength[i])
        else :
            ### print("ispec.size=",ispec.size,"wavelength.size=",wavelength.size)
            res=np.zeros((ispec.size,wavelength.size))
            for j,i in enumerate(ispec):
                res[j]=self.traceset.x_vs_wave(i,wavelength)
        return res

    def y(self,ispec=None,wavelength=None):
        """
        returns CCD y centroids for the spectra
        ispec can be None, scalar or a vector
        wavelength can be a vector but not allowing None #- similar as in specter.psf.PSF.y
        """
        if ispec is None :
            ispec = np.arange(self.traceset.nspec)
        else :
            ispec = np.atleast_1d(ispec)

        if wavelength is None :
            wavelength = self.wavelength(ispec)
        else :
            wavelength = np.atleast_1d(wavelength)

        if len(wavelength.shape)==2 :
            res=np.zeros(wavelength.shape)
            for j,i in enumerate(ispec):
                res[j]=self.traceset.y_vs_wave(ii,wavelength[i])
        else :
            res=np.zeros((ispec.size,wavelength.size))
            for j,i in enumerate(ispec):
                res[j]=self.traceset.y_vs_wave(i,wavelength)
        return res


    def wavelength(self,ispec=None,y=None):
        """
        returns wavelength evaluated at y
        """
        if y is None:
            y=np.arange(0,self.npix_y)
        else :
            y = np.atleast_1d(y)

        if ispec is None:
            ispec = np.arange(self.traceset.nspec)

        if np.size(ispec)==1 :
            return self.traceset.wave_vs_y(ispec,y)
        else :
            if np.size(y)==1 :
                res=np.zeros((ispec.size))
                for j,i in enumerate(ispec):
                    res[j]=self.traceset.wave_vs_y(i,y)
                return res
            else :
                res=np.zeros((ispec.size,y.size))
                for j,i in enumerate(ispec):
                    res[j]=self.traceset.wave_vs_y(i,y)
                return res

    def xsigma(self,ispec,wave):
        return self.traceset.xsig_vs_wave(ispec,wave)

    def ysigma(self,ispec,wave):
        return self.traceset.ysig_vs_wave(ispec,wave)

    def angstroms_per_pixel(self, ispec, wavelength):
        """
        Return CCD pixel width in Angstroms for spectrum ispec at given
        wavlength(s).  Wavelength may be scalar or array.
        """
        ww = self.wavelength(ispec, y=np.arange(self.npix_y))
        dw = np.gradient( ww )
        return np.interp(wavelength, ww, dw)
