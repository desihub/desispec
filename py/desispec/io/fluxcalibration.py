"""
io routines for flux calibration

"""
import os
from astropy.io import fits


# this is really temporary
# the idea is to have a datamodel for calibration stars spectra
def read_stellar_models(filename) :

    """
    read stellar models
    """
    flux=fits.getdata(filename, 0).astype('float64')
    wave=fits.getdata(filename, 1).astype('float64')
    fibers=fits.getdata(filename, 2).astype(int)
    return flux,wave,fibers


def write_flux_calibration(outfile,head,calibration, calibration_ivar, mask, convolved_calibration, convolved_calibration_ivar,wave) :
    """
    writes  flux calibration 
    """
    hdr = head
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
    calibration=fits.getdata(filename, 0).astype('float64')
    calib_ivar=fits.getdata(filename, "IVAR").astype('float64')
    mask=fits.getdata(filename, "MASK").astype('int') # ??? SOMEONE CHECK THIS ???
    convolved_calibration=fits.getdata(filename, "CCALIB").astype('float64')
    convolved_calib_ivar=fits.getdata(filename, "CIVAR").astype('float64')
    wave=fits.getdata(filename, "WAVELENGTH").astype('float64')
    
    return calibration,calib_ivar,mask,convolved_calibration,convolved_calib_ivar,wave
