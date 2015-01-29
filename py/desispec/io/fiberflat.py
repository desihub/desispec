"""
io routines for fiberflat

"""
import os
from astropy.io import fits


def write_fiberflat(outfile,head,fiberflat,fiberflat_ivar,fiberflat_mask,mean_spectrum,wave) :
    """
    write fiberflat
    """
    hdr = head
    hdr['EXTNAME'] = ('FIBERFLAT', 'no dimension')
    fits.writeto(outfile,fiberflat,header=hdr, clobber=True)
    
    hdr['EXTNAME'] = ('IVAR', 'no dimension')
    hdu = fits.ImageHDU(fiberflat_ivar, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('MASK', 'no dimension')
    hdu = fits.ImageHDU(fiberflat_mask, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('MEANSPEC', 'electrons')
    hdu = fits.ImageHDU(mean_spectrum, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('WAVELENGTH', '[Angstroms]')
    hdu = fits.ImageHDU(wave, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    
def read_fiberflat(filename) :
    """
    read fiberflat
    """
    fiberflat=fits.getdata(filename, 0).astype('float64')
    ivar=fits.getdata(filename, "IVAR").astype('float64')
    mask=fits.getdata(filename, "MASK").astype('int') # ??? SOMEONE CHECK THIS ???
    meanspec=fits.getdata(filename, "MEANSPEC").astype('float64')
    wave=fits.getdata(filename, "WAVELENGTH").astype('float64')
    
    return fiberflat,ivar,mask,meanspec,wave
    
