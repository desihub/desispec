"""
desispec.io.fluxcalibration
===========================

IO routines for flux calibration.
"""
from __future__ import absolute_import, print_function
import os
import time
from astropy.io import fits
from astropy.io.fits.convenience import table_to_hdu
from astropy.table import Table
import numpy,scipy

from desiutil.depend import add_dependencies
from desiutil.log import get_logger
from desiutil.io import encode_table

from .util import fitsheader, native_endian, makepath, checkgzip
from . import iotime

def write_stdstar_models(norm_modelfile, normalizedFlux, wave, fibers, data,
        fibermap, input_frames, header=None):
    """Writes the normalized flux for the best models.

    Args:
        norm_modelfile : output file path
        normalizedFlux : 2D array of flux[nstdstars, nwave]
        wave : 1D array of wavelengths[nwave] in Angstroms
        fibers : 1D array of fiberids for these spectra
        data : meta data table about which templates best fit
        fibermap : fibermaps rows for the input standard stars
        input_frames : Table with NIGHT, EXPID, CAMERA of input frames used
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

    fmhdu = table_to_hdu(Table(fibermap))
    fmhdu.name = 'FIBERMAP'
    hdulist.append(fmhdu)

    inhdu = table_to_hdu(Table(input_frames))
    inhdu.name = 'INPUT_FRAMES'
    hdulist.append(inhdu)

    t0 = time.time()
    tmpfile = norm_modelfile+".tmp"
    hdulist.writeto(tmpfile, overwrite=True, checksum=True)
    os.rename(tmpfile, norm_modelfile)
    duration = time.time() - t0
    log.info(iotime.format('write', norm_modelfile, duration))


def read_stdstar_models(filename):
    """Read stdstar models from filename.

    Args:
        filename (str): File containing standard star models.

    Returns:
        read_stdstar_models (tuple): flux[nspec, nwave], wave[nwave], fibers[nspec]
    """
    log = get_logger()
    t0 = time.time()
    filename = checkgzip(filename)
    with fits.open(filename, memmap=False) as fx:
        flux = native_endian(fx['FLUX'].data.astype('f8'))
        wave = native_endian(fx['WAVELENGTH'].data.astype('f8'))
        fibers = native_endian(fx['FIBERS'].data)
        metadata = fx['METADATA'].data

    duration = time.time() - t0
    log.info(iotime.format('read', filename, duration))

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

    if fluxcalib.fibercorr is not None :
        tbl = encode_table(fluxcalib.fibercorr)  #- unicode -> bytes
        tbl.meta['EXTNAME'] = 'FIBERCORR'
        hx.append( fits.convenience.table_to_hdu(tbl) )
        if fluxcalib.fibercorr_comments is not None : # add comments in header
            hdu=hx['FIBERCORR']
            for i in range(1,999):
                key = 'TTYPE'+str(i)
                if key in hdu.header:
                    value = hdu.header[key]
                    if value in fluxcalib.fibercorr_comments.keys() :
                        hdu.header[key] = (value, fluxcalib.fibercorr_comments[value])

    if fluxcalib.stdstar_fibermap is not None :
        fibermap = encode_table(fluxcalib.stdstar_fibermap)  #- unicode -> bytes
        fibermap.meta['EXTNAME'] = 'STDSTAR_FIBERMAP'
        hx.append( fits.convenience.table_to_hdu(fibermap) )

    t0 = time.time()
    hx.writeto(outfile+'.tmp', overwrite=True, checksum=True)
    os.rename(outfile+'.tmp', outfile)
    duration = time.time() - t0
    log.info(iotime.format('write', outfile, duration))

    return outfile


def read_flux_calibration(filename):
    """Read flux calibration file; returns a FluxCalib object
    """
    # Avoid a circular import conflict at package install/build_sphinx time.
    from ..fluxcalibration import FluxCalib
    log = get_logger()
    t0 = time.time()
    filename = checkgzip(filename)
    with fits.open(filename, memmap=False, uint=True) as fx:
        calib = native_endian(fx[0].data.astype('f8'))
        ivar = native_endian(fx["IVAR"].data.astype('f8'))
        mask = native_endian(fx["MASK"].data)
        wave = native_endian(fx["WAVELENGTH"].data.astype('f8'))
        header = fx[0].header

        if 'FIBERCORR' in fx:
            fibercorr = fx['FIBERCORR'].data
            # I need to open the header to read the comments
            fibercorr_comments = dict()
            head   = fx['FIBERCORR'].header
            for i in range(1,len(fibercorr.columns)+1) :
                k='TTYPE'+str(i)
                fibercorr_comments[head[k]]=head.comments[k]
        else:
            fibercorr = None
            fibercorr_comments = None

        if 'STDSTAR_FIBERMAP' in fx:
            stdstar_fibermap = fx['STDSTAR_FIBERMAP'].data
        else :
            stdstar_fibermap = None

    duration = time.time() - t0
    log.info(iotime.format('read', filename, duration))

    fluxcalib = FluxCalib(wave, calib, ivar, mask,
                          fibercorr=fibercorr, fibercorr_comments=fibercorr_comments,
                          stdstar_fibermap = stdstar_fibermap)
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
    # AR wave-dependent FIBER_FRACFLUX curve for the median seeing and median FIBER_FRACFLUX of the used exposures
    if averagefluxcalib.ffracflux_wave is not None:
        hx.append( fits.ImageHDU(averagefluxcalib.ffracflux_wave.astype('f4'), name='FFRACFLUX_WAVE') )
        hx[-1].header['MDSEEING'] = averagefluxcalib.median_seeing
        hx[-1].header['MDFFRACF'] = averagefluxcalib.median_ffracflux
        hx[-1].header['FACWPOW'] = averagefluxcalib.fac_wave_power
        hx[-1].header['FSTNIGHT'] = averagefluxcalib.first_night

    t0 = time.time()
    hx.writeto(outfile+'.tmp', overwrite=True, checksum=True)
    os.rename(outfile+'.tmp', outfile)
    duration = time.time() - t0
    log.info(iotime.format('write', outfile, duration))

    return outfile


def read_average_flux_calibration(filename):
    """Read average flux calibration file; returns an AverageFluxCalib object
    """

    # Avoid a circular import conflict at package install/build_sphinx time.
    from ..averagefluxcalibration import AverageFluxCalib
    log = get_logger()
    t0 = time.time()
    filename = checkgzip(filename)
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
        if "FFRACFLUX_WAVE" in fx:
            median_seeing          = fx["FFRACFLUX_WAVE"].header["MDSEEING"]
            median_ffracflux       = fx["FFRACFLUX_WAVE"].header["MDFFRACF"]
            fac_wave_power         = fx["FFRACFLUX_WAVE"].header["FACWPOW"]
            ffracflux_wave         = native_endian(fx["FFRACFLUX_WAVE"].data.astype('f8'))
            first_night            = fx["FFRACFLUX_WAVE"].header["FSTNIGHT"]
        else:
            median_seeing, median_ffracflux, fac_wave_power, ffracflux_wave = None, None, None, None
            first_night = None


    duration = time.time() - t0
    log.info(iotime.format('read', filename, duration))

    afluxcalib = AverageFluxCalib(wave=wave,
                                  average_calib=average_calib,
                                  atmospheric_extinction=atmospheric_extinction,
                                  seeing_term=seeing_term,
                                  pivot_airmass=pivot_airmass,
                                  pivot_seeing=pivot_seeing,
                                  atmospheric_extinction_uncertainty=atmospheric_extinction_uncertainty,
                                  seeing_term_uncertainty=seeing_term_uncertainty,
                                  median_seeing=median_seeing,
                                  median_ffracflux=median_ffracflux,
                                  fac_wave_power=fac_wave_power,
                                  ffracflux_wave=ffracflux_wave,
                                  first_night=first_night)

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
    stellarmodelfile = checkgzip(stellarmodelfile)
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
    duration = time.time() - t0
    log.info(iotime.format('read', stellarmodelfile, duration))

    return wavebins,fluxData,templateid,teff,logg,feh
