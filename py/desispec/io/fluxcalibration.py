"""
desispec.io.fluxcalibration
===========================

IO routines for flux calibration.
"""
from __future__ import absolute_import
import os
from astropy.io import fits
import numpy,scipy
from .util import fitsheader, native_endian, makepath

def write_stdstar_model(norm_modelfile,normalizedFlux,wave,fibers,data,header=None):
    """Writes the normalized flux for the best model.
    
    Args:
        norm_modelfile : output file path
        normalizedFlux : 2D array of flux[nstdstars, nwave]
        wave : 1D array of wavelengths[nwave] in Angstroms
        fibers : 1D array of fiberids for these spectra
        data : meta data table about which templates best fit; should include
            BESTMODELINDEX, TEMPLATEID, CHI2DOF
    """
    hdr = fitsheader(header)
    hdr['EXTNAME'] = ('FLUX', 'erg/s/cm2/A')
    hdr['BUNIT'] = ('erg/s/cm2/A', 'Flux units')
    hdu1=fits.PrimaryHDU(normalizedFlux,header=hdr)
    #fits.writeto(norm_modelfile,normalizedFlux,header=hdr, clobber=True)

    hdr['EXTNAME'] = ('WAVE', '[Angstroms]')
    hdr['BUNIT'] = ('Angstrom', 'Wavelength units')
    hdu2 = fits.ImageHDU(wave, header=hdr)

    hdr['EXTNAME'] = ('FIBERS', 'no dimension')
    hdu3 = fits.ImageHDU(fibers, header=hdr)

    hdr['EXTNAME'] = ('METADATA', 'no dimension')
    from astropy.io.fits import Column
    BESTMODELINDEX=Column(name='BESTMODELINDEX',format='K',array=data['BESTMODEL'])
    TEMPLATEID=Column(name='TEMPLATEID',format='K',array=data['TEMPLATEID'])
    CHI2DOF=Column(name='CHI2DOF',format='D',array=data['CHI2DOF'])
    cols=fits.ColDefs([BESTMODELINDEX,TEMPLATEID,CHI2DOF])
    tbhdu=fits.BinTableHDU.from_columns(cols,header=hdr)

    hdulist=fits.HDUList([hdu1,hdu2,hdu3,tbhdu])
    hdulist.writeto(norm_modelfile,clobber=True)
    #fits.append(norm_modelfile,cols,header=tbhdu.header)

def read_stdstar_models(filename):
    """Read stdstar models from filename.

    Args:
        filename (str): File containing standard star models.

    Returns:
        read_stdstar_models (tuple): flux[nspec, nwave], wave[nwave], fibers[nspec]
    """
    flux = native_endian(fits.getdata(filename, 0))
    wave = native_endian(fits.getdata(filename, 1))
    fibers = native_endian(fits.getdata(filename, 2))
    return flux,wave,fibers


def write_flux_calibration(outfile, fluxcalib, header=None):
    """Writes  flux calibration.
    
    Args:
        outfile : output file name
        fluxcalib : FluxCalib object
        
    Options:
        header : dict-like object of key/value pairs to include in header
    """
    hdr = fitsheader(header)
    hdr['EXTNAME'] = ('FLUXCALIB', 'CHECK UNIT')
    fits.writeto(outfile,fluxcalib.calib,header=hdr, clobber=True)

    hdr['EXTNAME'] = ('IVAR', 'CHECK UNIT')
    hdu = fits.ImageHDU(fluxcalib.ivar, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)

    hdr['EXTNAME'] = ('MASK', 'no dimension')
    hdu = fits.ImageHDU(fluxcalib.mask, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)

    hdr['EXTNAME'] = ('WAVELENGTH', '[Angstroms]')
    hdu = fits.ImageHDU(fluxcalib.wave, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)

def read_flux_calibration(filename):
    """Read flux calibration file; returns a FluxCalib object
    """
    # Avoid a circular import conflict at package install/build_sphinx time.
    from ..fluxcalibration import FluxCalib
    calib=native_endian(fits.getdata(filename, 0))
    ivar=native_endian(fits.getdata(filename, "IVAR"))
    mask=native_endian(fits.getdata(filename, "MASK", uint=True))
    wave=native_endian(fits.getdata(filename, "WAVELENGTH"))

    fluxcalib = FluxCalib(wave, calib, ivar, mask)
    fluxcalib.header = fits.getheader(filename, 0)
    return fluxcalib


def read_stdstar_templates(stellarmodelfile):
    """
    Reads an input stellar model file
    
    Args:
        stellarmodelfile : input filename
    
    Returns (wave, flux, templateid) tuple:
        wave : 1D[nwave] array of wavelengths [Angstroms]
        flux : 2D[nmodel, nwave] array of model fluxes
        templateid : 1D[nmodel] array of template IDs for each spectrum
    """
    phdu=fits.open(stellarmodelfile)
    hdr0=phdu[0].header
    crpix1=hdr0['CRPIX1']
    crval1=hdr0['CRVAL1']
    cdelt1=hdr0['CDELT1']
    if hdr0["LOGLAM"]==1: #log bins
        wavebins=10**(crval1+cdelt1*numpy.arange(len(phdu[0].data[0])))
    else: #lin bins
        model_wave_step   = cdelt1
        model_wave_offset = (crval1-cdelt1*(crpix1-1))
        wavebins=model_wave_step*numpy.arange(n_model_wave) + model_wave_offset
    paramData=phdu[1].data
    templateid=paramData["TEMPLATEID"]
    fluxData=phdu[0].data

    phdu.close()

    return wavebins,fluxData,templateid
