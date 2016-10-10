"""
desispec.io.fluxcalibration
===========================

IO routines for flux calibration.
"""
from __future__ import absolute_import, print_function
import os
from astropy.io import fits
import numpy,scipy

from desiutil.depend import add_dependencies

from .util import fitsheader, native_endian, makepath

def write_stdstar_models(norm_modelfile,normalizedFlux,wave,fibers,data,header=None):
    """Writes the normalized flux for the best models.
    
    Args:
        norm_modelfile : output file path
        normalizedFlux : 2D array of flux[nstdstars, nwave]
        wave : 1D array of wavelengths[nwave] in Angstroms
        fibers : 1D array of fiberids for these spectra
        data : meta data table about which templates best fit; should include
            BESTMODEL, TEMPLATEID, CHI2DOF, REDSHIFT
    """
    hdr = fitsheader(header)
    add_dependencies(hdr)
    
    hdr['EXTNAME'] = ('FLUX', 'erg/s/cm2/A')
    hdr['BUNIT'] = ('erg/s/cm2/A', 'Flux units')
    hdu1=fits.PrimaryHDU(normalizedFlux.astype('f4'), header=hdr.copy())

    hdr['EXTNAME'] = ('WAVELENGTH', '[Angstroms]')
    hdr['BUNIT'] = ('Angstrom', 'Wavelength units')
    hdu2 = fits.ImageHDU(wave.astype('f4'), header=hdr.copy())

    hdr['EXTNAME'] = ('FIBERS', 'no dimension')
    hdu3 = fits.ImageHDU(fibers, header=hdr.copy())

    hdr['EXTNAME'] = ('METADATA', 'no dimension')
    from astropy.io.fits import Column
    BESTMODEL=Column(name='BESTMODEL',format='K',array=data['BESTMODEL'])
    TEMPLATEID=Column(name='TEMPLATEID',format='K',array=data['TEMPLATEID'])
    CHI2DOF=Column(name='CHI2DOF',format='D',array=data['CHI2DOF'])
    REDSHIFT=Column(name='REDSHIFT',format='D',array=data['REDSHIFT'])
    cols=fits.ColDefs([BESTMODEL,TEMPLATEID,CHI2DOF,REDSHIFT])
    tbhdu=fits.BinTableHDU.from_columns(cols,header=hdr)

    hdulist=fits.HDUList([hdu1,hdu2,hdu3,tbhdu])
    tmpfile = norm_modelfile+".tmp"
    hdulist.writeto(tmpfile, clobber=True, checksum=True)
    os.rename(tmpfile, norm_modelfile)
    #fits.append(norm_modelfile,cols,header=tbhdu.header)

def read_stdstar_models(filename):
    """Read stdstar models from filename.

    Args:
        filename (str): File containing standard star models.

    Returns:
        read_stdstar_models (tuple): flux[nspec, nwave], wave[nwave], fibers[nspec]
    """
    with fits.open(filename, memmap=False) as fx:
        flux = native_endian(fx['FLUX'].data.astype('f8'))
        wave = native_endian(fx['WAVELENGTH'].data.astype('f8'))
        fibers = native_endian(fx['FIBERS'].data)
    
    return flux, wave, fibers


def write_flux_calibration(outfile, fluxcalib, header=None):
    """Writes  flux calibration.
    
    Args:
        outfile : output file name
        fluxcalib : FluxCalib object
        
    Options:
        header : dict-like object of key/value pairs to include in header
    """
    hx = fits.HDUList()
    
    hdr = fitsheader(header)
    add_dependencies(hdr)
    
    hdr['EXTNAME'] = 'FLUXCALIB'
    hdr['BUNIT'] = ('(electrons/A) / (1e-17 erg/s/cm2/A)', 'electrons per flux unit')
    hx.append( fits.PrimaryHDU(fluxcalib.calib.astype('f4'), header=hdr) )
    hx.append( fits.ImageHDU(fluxcalib.ivar.astype('f4'), name='IVAR') )
    hx.append( fits.CompImageHDU(fluxcalib.mask, name='MASK') )
    hx.append( fits.ImageHDU(fluxcalib.wave, name='WAVELENGTH') )
    
    hx.writeto(outfile+'.tmp', clobber=True, checksum=True)
    os.rename(outfile+'.tmp', outfile)

    return outfile

def read_flux_calibration(filename):
    """Read flux calibration file; returns a FluxCalib object
    """
    # Avoid a circular import conflict at package install/build_sphinx time.
    from ..fluxcalibration import FluxCalib
    fx = fits.open(filename, memmap=False, uint=True)
    calib = native_endian(fx[0].data.astype('f8'))
    ivar = native_endian(fx["IVAR"].data.astype('f8'))
    mask = native_endian(fx["MASK"].data)
    wave = native_endian(fx["WAVELENGTH"].data.astype('f8'))

    fluxcalib = FluxCalib(wave, calib, ivar, mask)
    fluxcalib.header = fx[0].header
    fx.close()
    return fluxcalib


def read_stdstar_templates(stellarmodelfile):
    """
    Reads an input stellar model file
    
    Args:
        stellarmodelfile : input filename
    
    Returns (wave, flux, templateid, teff, logg, feh) tuple:
        wave : 1D[nwave] array of wavelengths [Angstroms]
        flux : 2D[nmodel, nwave] array of model fluxes
        templateid : 1D[nmodel] array of template IDs for each spectrum
        teff : 1D[nmodel] array of effective temperature for each model
        logg : 1D[nmodel] array of surface gravity for each model
        feh : 1D[nmodel] array of metallicity for each model
    """
    phdu=fits.open(stellarmodelfile, memmap=False)
    
    #- New templates have wavelength in HDU 2
    if len(phdu) >= 3:
        wavebins = native_endian(phdu[2].data)
    #- Old templates define wavelength grid in HDU 0 keywords
    else:        
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
    teff=paramData["TEFF"]
    logg=paramData["LOGG"]
    feh=paramData["FEH"]
    fluxData=native_endian(phdu[0].data)

    phdu.close()

    return wavebins,fluxData,templateid,teff,logg,feh
