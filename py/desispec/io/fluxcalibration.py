"""
desispec.io.fluxcalibration
===========================

IO routines for flux calibration.
"""
from __future__ import absolute_import, print_function
import os
import time
from astropy.io import fits
from astropy.table import Table
import numpy,scipy

from desiutil.depend import add_dependencies
from desiutil.log import get_logger

from .util import fitsheader, native_endian, makepath, iotime_message

def write_stdstar_models(norm_modelfile,normalizedFlux,wave,fibers,data,header=None):
    """Writes the normalized flux for the best models.

    Args:
        norm_modelfile : output file path
        normalizedFlux : 2D array of flux[nstdstars, nwave]
        wave : 1D array of wavelengths[nwave] in Angstroms
        fibers : 1D array of fiberids for these spectra
        data : meta data table about which templates best fit
    """
    log = get_logger()
    hdr = fitsheader(header)
    add_dependencies(hdr)

    #- support input Table, np.array, and dict
    data = Table(data)

    hdr['EXTNAME'] = ('FLUX', '[10**-17 erg/(s cm2 Angstrom)]')
    hdr['BUNIT'] = ('10**-17 erg/(s cm2 Angstrom)', 'Flux units')
    hdu1=fits.PrimaryHDU(normalizedFlux.astype('f4'), header=hdr)

    hdu2 = fits.ImageHDU(wave.astype('f4'))
    hdu2.header['EXTNAME'] = ('WAVELENGTH', '[Angstrom]')
    hdu2.header['BUNIT'] = ('Angstrom', 'Wavelength units')

    hdu3 = fits.ImageHDU(fibers, name='FIBERS')

    # metadata
    from astropy.io.fits import Column
    cols=[]
    for k in data.colnames:
        if len(data[k].shape)==1 :
            cols.append(Column(name=k,format='D',array=data[k]))
    tbhdu=fits.BinTableHDU.from_columns(fits.ColDefs(cols), name='METADATA')

    hdulist=fits.HDUList([hdu1,hdu2,hdu3,tbhdu])

    # add coefficients
    if "COEFF" in data.colnames:
        hdulist.append(fits.ImageHDU(data["COEFF"],name="COEFF"))

    t0 = time.time()
    tmpfile = norm_modelfile+".tmp"
    hdulist.writeto(tmpfile, overwrite=True, checksum=True)
    os.rename(tmpfile, norm_modelfile)
    iotime = time.time() - t0
    log.info(iotime_message('write', norm_modelfile, iotime))


def read_stdstar_models(filename):
    """Read stdstar models from filename.

    Args:
        filename (str): File containing standard star models.

    Returns:
        read_stdstar_models (tuple): flux[nspec, nwave], wave[nwave], fibers[nspec]
    """
    log = get_logger()
    t0 = time.time()
    with fits.open(filename, memmap=False) as fx:
        flux = native_endian(fx['FLUX'].data.astype('f8'))
        wave = native_endian(fx['WAVELENGTH'].data.astype('f8'))
        fibers = native_endian(fx['FIBERS'].data)
        metadata = fx['METADATA'].data

    iotime = time.time() - t0
    log.info(iotime_message('read', filename, iotime))

    return flux, wave, fibers, metadata


def write_flux_calibration(outfile, fluxcalib, header=None):
    """Writes  flux calibration.

    Args:
        outfile : output file name
        fluxcalib : FluxCalib object

    Options:
        header : dict-like object of key/value pairs to include in header
    """
    log = get_logger()
    hx = fits.HDUList()

    hdr = fitsheader(header)
    add_dependencies(hdr)

    hdr['EXTNAME'] = 'FLUXCALIB'
    hdr['BUNIT'] = ('10**+17 cm2 count s / erg', 'i.e. (elec/A) / (1e-17 erg/s/cm2/A)')
    hx.append( fits.PrimaryHDU(fluxcalib.calib.astype('f4'), header=hdr) )
    hx.append( fits.ImageHDU(fluxcalib.ivar.astype('f4'), name='IVAR') )
    # hx.append( fits.CompImageHDU(fluxcalib.mask, name='MASK') )
    hx.append( fits.ImageHDU(fluxcalib.mask, name='MASK') )
    hx.append( fits.ImageHDU(fluxcalib.wave.astype('f4'), name='WAVELENGTH') )
    hx[-1].header['BUNIT'] = 'Angstrom'

    t0 = time.time()
    hx.writeto(outfile+'.tmp', overwrite=True, checksum=True)
    os.rename(outfile+'.tmp', outfile)
    iotime = time.time() - t0
    log.info(iotime_message('write', outfile, iotime))

    return outfile


def read_flux_calibration(filename):
    """Read flux calibration file; returns a FluxCalib object
    """
    # Avoid a circular import conflict at package install/build_sphinx time.
    from ..fluxcalibration import FluxCalib
    log = get_logger()
    t0 = time.time()
    with fits.open(filename, memmap=False, uint=True) as fx:
        calib = native_endian(fx[0].data.astype('f8'))
        ivar = native_endian(fx["IVAR"].data.astype('f8'))
        mask = native_endian(fx["MASK"].data)
        wave = native_endian(fx["WAVELENGTH"].data.astype('f8'))
        header = fx[0].header

    iotime = time.time() - t0
    log.info(iotime_message('read', filename, iotime))

    fluxcalib = FluxCalib(wave, calib, ivar, mask)
    fluxcalib.header = header

    return fluxcalib


def write_average_flux_calibration(outfile, averagefluxcalib):
    """Writes average flux calibration.

    Args:
        outfile : output file name
        averagefluxcalib : AverageFluxCalib object

    Options:
        header : dict-like object of key/value pairs to include in header
    """
    log = get_logger()
    hx = fits.HDUList()
    hx.append( fits.PrimaryHDU(averagefluxcalib.average_calib.astype('f4')) )
    hx[-1].header['EXTNAME'] = 'FLUXCALIB'
    hx[-1].header['BUNIT'] = ('10**+17 cm2 count / erg', 'i.e. (elec/A/s) / (1e-17 erg/s/cm2/A)')
    hx.append( fits.ImageHDU(averagefluxcalib.atmospheric_extinction.astype('f4'), name='ATERM') )
    hx[-1].header['PAIRMASS'] = averagefluxcalib.pivot_airmass
    hx.append( fits.ImageHDU(averagefluxcalib.seeing_term.astype('f4'), name='STERM') )
    hx[-1].header['PSEEING'] = averagefluxcalib.pivot_seeing
    hx.append( fits.ImageHDU(averagefluxcalib.wave.astype('f4'), name='WAVELENGTH') )
    hx[-1].header['BUNIT'] = 'Angstrom'
    if averagefluxcalib.atmospheric_extinction_uncertainty is not None :
      hx.append( fits.ImageHDU(averagefluxcalib.atmospheric_extinction_uncertainty.astype('f4'), name='ATERM_ERR') )
    if averagefluxcalib.seeing_term_uncertainty is not None :
        hx.append( fits.ImageHDU(averagefluxcalib.seeing_term_uncertainty.astype('f4'), name='STERM_ERR') )

    t0 = time.time()
    hx.writeto(outfile+'.tmp', overwrite=True, checksum=True)
    os.rename(outfile+'.tmp', outfile)
    iotime = time.time() - t0
    log.info(iotime_message('write', outfile, iotime))

    return outfile


def read_average_flux_calibration(filename):
    """Read average flux calibration file; returns an AverageFluxCalib object
    """
    
    # Avoid a circular import conflict at package install/build_sphinx time.
    from ..averagefluxcalibration import AverageFluxCalib
    log = get_logger()
    t0 = time.time()
    with fits.open(filename, memmap=False, uint=True) as fx:
        average_calib = native_endian(fx[0].data.astype('f8'))
        atmospheric_extinction = native_endian(fx["ATERM"].data.astype('f8'))
        seeing_term            = native_endian(fx["STERM"].data.astype('f8'))
        pivot_airmass          = fx["ATERM"].header["PAIRMASS"]
        pivot_seeing           = fx["STERM"].header["PSEEING"]
        wave                   = native_endian(fx["WAVELENGTH"].data.astype('f8'))
        if "ATERM_ERR" in fx :
            atmospheric_extinction_uncertainty = native_endian(fx["ATERM_ERR"].data.astype('f8'))
        else :
            atmospheric_extinction_uncertainty = None
        if "STERM_ERR" in fx :
            seeing_term_uncertainty = native_endian(fx["STERM_ERR"].data.astype('f8'))
        else :
            seeing_term_uncertainty = None

    iotime = time.time() - t0
    log.info(iotime_message('read', filename, iotime))

    afluxcalib = AverageFluxCalib(wave=wave,
                                  average_calib=average_calib,
                                  atmospheric_extinction=atmospheric_extinction,
                                  seeing_term=seeing_term,
                                  pivot_airmass=pivot_airmass,
                                  pivot_seeing=pivot_seeing,
                                  atmospheric_extinction_uncertainty=atmospheric_extinction_uncertainty,
                                  seeing_term_uncertainty=seeing_term_uncertainty)

    return afluxcalib


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
    log = get_logger()
    t0 = time.time()
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
            n_model_wave = phdu[0].data.shape[1]
            wavebins=model_wave_step*numpy.arange(n_model_wave) + model_wave_offset

    paramData=phdu[1].data
    templateid=paramData["TEMPLATEID"]
    teff=paramData["TEFF"]
    logg=paramData["LOGG"]
    feh=paramData["FEH"]
    fluxData=native_endian(phdu[0].data)

    phdu.close()
    iotime = time.time() - t0
    log.info(iotime_message('read', stellarmodelfile, iotime))

    return wavebins,fluxData,templateid,teff,logg,feh
