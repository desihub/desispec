#!/usr/bin/env python

"""
Given a psf output file eg. output from bootcalib.write_psf or desimodel/data/specpsf/PSF* files
this defines an interface that other codes can use the trace and wavelength solutions

Mostly making parallel to specter.psf.PSF baseclass and inheriting as needed, but only xtrace,
ytrace and wavelength solution available for this case. No resolution information yet.
"""

import numbers
import numpy as np
from desiutil import funcfits as dufits
from numpy.polynomial.legendre import Legendre,legval,legfit
import astropy.io.fits as fits
import scipy.optimize


class PSF(object):
    """
    Base class for 2D psf
    """
    def __init__(self,filename):
        """
        load header, xcoeff, ycoeff from the file
        filename should have HDU1: Xcoeff, HDU2: Ycoeff
        """
        # psfdata = fits.open(filename, memmap=False)
        with fits.open(filename, memmap=False) as psfdata:
            xcoeff=psfdata[0].data
            hdr=psfdata[0].header
            wmin=hdr['WAVEMIN']
            wmax=hdr['WAVEMAX']
            ycoeff=psfdata[1].data
        
            arm = hdr['CAMERA'].lower()[0]
            npix_x = hdr['NPIX_X']
            npix_y = hdr['NPIX_Y']

            if 'XSIGMA' in psfdata:
                self.xsigma_boot=psfdata['XSIGMA'].data
            if 'WSIGMA' in psfdata: #- w sigma legendre expansion coefficients
                self.wcoeff=psfdata['WSIGMA'].data
        
            if arm not in ['b','r','z']:
                raise ValueError("arm not in b, r, or z. File should be of the form psfboot-r0.fits.")  
            #- Get the coeffiecients
            nspec=xcoeff.shape[0]
            ncoeff=xcoeff.shape[1]
        
            self.npix_x=npix_x
            self.npix_y=npix_y
            self.xcoeff=xcoeff
            self.ycoeff=ycoeff
            self.wmin=wmin
            self.wmax=wmax
            self.nspec=nspec
            self.ncoeff=ncoeff
            #invertion should be done at psf creation time and saved into file
            c,ymin,ymax=self.invert(coeff=self.ycoeff)
            self.icoeff=c
            self.ymin=ymin
            self.ymax=ymax

    def invert(self, domain=None, coeff=None, deg=None):
        """
        Utility to return a traceset modeling x vs. y instead of y vs. x
        """
        if domain is None:
            domain=[self.wmin,self.wmax]
        ispec=np.arange(self.nspec) # Doing for all spectra
        if coeff is None:
            coeff=self.ycoeff # doing y-wavelength map
        ytmp=list()
        for ii in ispec:
            fit_dict=dufits.mk_fit_dict(coeff[ii,:],coeff.shape[1],'legendre',domain[0],domain[1])
            xtmp=np.array((domain[0],domain[1]))
            yfit = dufits.func_val(xtmp, fit_dict)
            ytmp.append(yfit)

        ymin = np.min(ytmp)
        ymax = np.max(ytmp)
        x = np.linspace(domain[0], domain[1], 1000)
        if deg is None:
            deg = self.ncoeff+2

        #- Now get the coefficients for inverse mapping    
        c = np.zeros((coeff.shape[0], deg+1))
        for ii in ispec:
            fit_dict=dufits.mk_fit_dict(coeff[ii,:],coeff.shape,'legendre',domain[0],domain[1])
            y = dufits.func_val(x,fit_dict)
            yy = 2.0 * (y-ymin) / (ymax-ymin) - 1.0
            c[ii] = legfit(yy, x, deg)
            
        return c,ymin,ymax

    def x(self,ispec=None,wavelength=None):
        """
        returns CCD x centroids for the spectra
        ispec can be None, scalar or a vector
        wavelength can be None or a vector
        """
        if wavelength is None:
            #- ispec = None -> all the spectra
            if ispec is None:
                ispec=np.arange(self.nspec)
                x=list()
                #x=np.array((len(ispec),len(wavelength)))
                for ii in ispec:
                    wave=self.wavelength(ii)
                    fit_dictx=dufits.mk_fit_dict(self.xcoeff[ii],self.ncoeff,'legendre',self.wmin,self.wmax)
                    xfit=dufits.func_val(wave,fit_dictx)
                    x.append(xfit)
                return np.array(x)

            if isinstance(ispec,(np.ndarray,list,tuple)):
                x=list()
                for ii in ispec:
                    wave=self.wavelength(ii)
                    fit_dictx=dufits.mk_fit_dict(self.xcoeff[ii],self.ncoeff,'legendre',self.wmin,self.wmax)
                    xfit=dufits.func_val(wave,fit_dictx)
                    x.append(xfit)
                return np.array(x)
    
            else: # int ispec
                wave=self.wavelength(ispec)
                fit_dictx=dufits.mk_fit_dict(self.xcoeff[ispec],self.ncoeff,'legendre',self.wmin,self.wmax)
                x=dufits.func_val(wave,fit_dictx)
                return np.array(x)
        
        #- wavelength not None but a scalar or 1D-vector here and below
        wavelength = np.asarray(wavelength)
        if isinstance(ispec, numbers.Integral):
            fit_dictx=dufits.mk_fit_dict(self.xcoeff[ispec],self.ncoeff,'legendre',self.wmin,self.wmax)
            x=dufits.func_val(wavelength,fit_dictx)
            return np.array(x)
        if ispec is None:
            ispec=np.arange(self.nspec) 
        x=list()
        for ii in ispec: #- for a None or a np.ndarray or anything that can be iterated case
            fit_dictx=dufits.mk_fit_dict(self.xcoeff[ii],self.ncoeff,'legendre',self.wmin,self.wmax)
            xfit=dufits.func_val(wavelength,fit_dictx)
            x.append(xfit)
        return np.array(x)


    def y(self,ispec=None,wavelength=None):
        """
        returns CCD y centroids for the spectra
        ispec can be None, scalar or a vector
        wavelength can be a vector but not allowing None #- similar as in specter.psf.PSF.y
        """
        if wavelength is None:
            raise ValueError("PSF.y requires wavelength 1D vector")
            
        wavelength = np.asarray(wavelength)
        if ispec is None:
            ispec=np.arange(self.nspec)
            y=list()
            for ii in ispec:
                fit_dicty=dufits.mk_fit_dict(self.ycoeff[ii],self.ncoeff,'legendre',self.wmin,self.wmax)
                yfit=dufits.func_val(wavelength,fit_dicty)
                y.append(yfit) 
            return np.array(y)

        if isinstance(ispec,(np.ndarray,list,tuple)):
            y=list()
            for ii in ispec:
                fit_dicty=dufits.mk_fit_dict(self.ycoeff[ii],self.ncoeff,'legendre',self.wmin,self.wmax)
                yfit=dufits.func_val(wavelength,fit_dicty)
                y.append(yfit)
            return np.array(y)

        if isinstance(ispec, numbers.Integral): # int ispec
            fit_dicty=dufits.mk_fit_dict(self.ycoeff[ispec],self.ncoeff,'legendre',self.wmin,self.wmax)
            y=dufits.func_val(wavelength,fit_dicty)
            return np.array(y)
    
    def wavelength(self,ispec=None,y=None):
        """
        returns wavelength evaluated at y
        """
        if y is None:
            y=np.arange(0,self.npix_y)
        if ispec is None:
            ispec=np.arange(self.nspec)
        c=self.icoeff
        ymin=self.ymin
        ymax=self.ymax

        if isinstance(ispec, numbers.Integral):
            new_dict=dufits.mk_fit_dict(c[ispec,:],c[ispec,:].shape,'legendre',ymin,ymax)
            wfit=dufits.func_val(y,new_dict)
            return wfit
        else:
            ww=list()
            for ii in ispec:
                new_dict=dufits.mk_fit_dict(c[ii,:],c[ii,:].shape,'legendre',ymin,ymax)
                wfit=dufits.func_val(y,new_dict)
                ww.append(wfit)
        return np.array(ww)

    def xsigma(self,ispec,wave):
        if hasattr(self,'xsigma_boot'):
            if np.isscalar(wave):
                return self.xsigma_boot[ispec]
            else:
                return np.full(len(wave),self.xsigma_boot[ispec]) #- constant xsigma for a given fiber

    def wdisp(self,ispec,wave):
        #- wave: scalar or vector, ispec: scalar integer TODO: make useful for other permutations
        if hasattr(self,'wcoeff'):
            new_dict=dufits.mk_fit_dict(self.wcoeff[ispec],self.wcoeff.shape[1],'legendre',self.wmin,self.wmax)
            wsigma=dufits.func_val(wave,new_dict)
            return wsigma

    def angstroms_per_pixel(self, ispec, wavelength):
        """
        Return CCD pixel width in Angstroms for spectrum ispec at given
        wavlength(s).  Wavelength may be scalar or array.
        """
        ww = self.wavelength(ispec, y=np.arange(self.npix_y))
        dw = np.gradient( ww )
        return np.interp(wavelength, ww, dw)
