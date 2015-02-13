"""
io routines for flux calibration

"""
import os
from astropy.io import fits
from desispec.io.util import fitsheader, native_endian, makepath

# this is really temporary
# the idea is to have a datamodel for calibration stars spectra
def read_stellar_models(filename) :
    """
    read stellar models from filename
    
    returns flux[nspec, nwave], wave[nwave], fibers[nspec]
    """
    flux = native_endian(fits.getdata(filename, 0))
    wave = native_endian(fits.getdata(filename, 1))
    fibers = native_endian(fits.getdata(filename, 2))
    return flux,wave,fibers


def write_flux_calibration(outfile,calibration, calibration_ivar, mask, convolved_calibration, convolved_calibration_ivar,wave,header=None):
    """
    writes  flux calibration 
    """
    hdr = fitsheader(header)
    hdr['EXTNAME'] = ('CALIB', 'CHECK UNIT')
    fits.writeto(outfile,calibration,header=hdr, clobber=True)
    
    hdr['EXTNAME'] = ('IVAR', 'CHECK UNIT')
    hdu = fits.ImageHDU(calibration_ivar, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('MASK', 'no dimension')
    hdu = fits.ImageHDU(mask, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    
    hdr['EXTNAME'] = ('CCALIB', 'CHECK UNIT')
    hdu = fits.ImageHDU(convolved_calibration, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('CIVAR', 'CHECK UNIT')
    hdu = fits.ImageHDU(convolved_calibration_ivar, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('WAVELENGTH', '[Angstroms]')
    hdu = fits.ImageHDU(wave, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
def read_flux_calibration(filename) :

    """
    read flux calibration
    """
    calibration=native_endian(fits.getdata(filename, 0))
    calib_ivar=native_endian(fits.getdata(filename, "IVAR"))
    mask=native_endian(fits.getdata(filename, "MASK"))
    convolved_calibration=native_endian(fits.getdata(filename, "CCALIB"))
    convolved_calib_ivar=native_endian(fits.getdata(filename, "CIVAR"))
    wave=native_endian(fits.getdata(filename, "WAVELENGTH"))
    
    return calibration,calib_ivar,mask,convolved_calibration,convolved_calib_ivar,wave
